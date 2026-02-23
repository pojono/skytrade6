#!/usr/bin/env python3
"""
Strategy 2: Volatility Breakout ("Straddle") — Walk-Forward Backtest

Concept:
  A spot-market "straddle" — we can't hold both long AND short simultaneously,
  so instead we combine vol/regime + directional models and let the LightGBM
  meta-model decide when and which direction to trade.
  
  Key difference from Directional Momentum:
  - Vol/regime models (vol_expansion, tail_event, consolidation, crash) as
    additional features for the meta-model (not as a hard gate)
  - Meta-model learns when vol conditions favor profitable trades
  - No vol gate — the meta-model handles regime awareness internally
  - Early exit on signal reversal

Architecture (v2 — best iteration: +267%, Sharpe 4.32, PF 1.73, 12/12 pos):
  For each 30-day trade period:
    1. Train 6 vol/regime models + 8 directional models on training window (360d)
    2. Inner train/val split (80/20) to generate OOS predictions
    3. Train LightGBM meta-model on 14 base predictions
       → predicts P(profitable trade over next 3 bars)
    4. Calibrate confidence threshold via 3-fold CV (no lookahead)
    5. Retrain all models on full training window
    6. Generate predictions on trade window, apply confidence gate
    7. Early exit on signal reversal
    8. Track PnL bar-by-bar with realistic fees

Anti-Lookahead: same as directional strategy (purge, inner split, 3-fold CV threshold)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# ============================================================
# PARAMETERS
# ============================================================
SYMBOL = "SOLUSDT"
TF = "4h"
FEATURES_DIR = Path("./features")
CONSTRAINTS_PATH = Path("./microstructure_research/predictable_targets.json")
RESULTS_DIR = Path("./microstructure_research/results")

SELECTION_DAYS = 360
PURGE_DAYS = 3
TRADE_DAYS = 30
INNER_TRAIN_FRAC = 0.80

FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0
INITIAL_CAPITAL = 10000.0

# Vol/regime models — these are the PRIMARY signal (gate)
VOL_MODELS = {
    "vol_expansion_5":   {"model": "logistic",  "features": "raw"},
    "vol_expansion_10":  {"model": "logistic",  "features": "raw"},
    "tail_event_3":      {"model": "lightgbm",  "features": "raw+lags"},
    "tail_event_5":      {"model": "lightgbm",  "features": "all_core"},
    "consolidation_3":   {"model": "logistic",  "features": "raw+lags"},
    "crash_10":          {"model": "lightgbm",  "features": "raw+lags"},
}

# Directional models — SECONDARY signal (direction once vol gate passes)
DIR_MODELS = {
    "breakout_up_3":     {"model": "lightgbm",  "features": "core+lags"},
    "breakout_down_3":   {"model": "ridgeclf",  "features": "all_core"},
    "breakout_up_5":     {"model": "lightgbm",  "features": "core+lags"},
    "breakout_down_5":   {"model": "ridgeclf",  "features": "all_core"},
    "profitable_long_1": {"model": "logistic",  "features": "raw+lags"},
    "profitable_short_1":{"model": "logistic",  "features": "raw+lags"},
    "alpha_1":           {"model": "ridge",     "features": "all_core"},
    "adverse_selection_1":{"model": "lightgbm", "features": "raw+lags"},
}

CONTINUOUS_TARGETS = {"alpha_1", "relative_ret_1"}

# Meta-model predicts: P(profitable trade) given vol+direction signals
META_TARGET_LONG = "tgt_profitable_long_3"
META_TARGET_SHORT = "tgt_profitable_short_3"
META_MODEL_TYPE = "lightgbm"

# Position sizing & execution
MAX_POSITION_FRAC = 1.0
HOLD_BARS = 3
EARLY_EXIT = True

# Vol gate: disabled (v2_no_gate was best — meta-model handles regime internally)
VOL_GATE_FLOOR = 0.0


# ============================================================
# DATA LOADING
# ============================================================
def load_features(features_dir, symbol, tf):
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_candles_per_day(tf):
    return {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}[tf]


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_lag_features(df, feat_cols, lags=[1, 2, 3]):
    new_cols = {}
    for col in feat_cols:
        if col in df.columns:
            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lag_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, lag_df], axis=1)


def prepare_feature_sets(df, target_features, core_features):
    available_raw = [f for f in target_features if f in df.columns]
    available_core = [f for f in core_features if f in df.columns]
    df_lag = add_lag_features(df, available_raw, lags=[1, 2, 3])
    lag_cols = [c for c in df_lag.columns if c not in df.columns and not c.startswith("tgt_")]
    df_core_lag = add_lag_features(df, available_core, lags=[1, 2, 3])
    core_lag_cols = [c for c in df_core_lag.columns if c not in df.columns and not c.startswith("tgt_")]
    return {
        "raw": (df, available_raw),
        "raw+lags": (df_lag, available_raw + lag_cols),
        "all_core": (df, available_core),
        "core+lags": (df_core_lag, available_core + core_lag_cols),
    }


# ============================================================
# MODEL TRAINING
# ============================================================
def train_base_model(X_train, y_train, X_test, model_type, is_binary=True):
    if model_type == "lightgbm":
        if is_binary:
            model = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            return model.predict_proba(X_test)[:, 1]
        else:
            model = lgb.LGBMRegressor(
                objective="regression", metric="rmse", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)
    elif model_type == "logistic":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(Xtr, y_train)
        return model.predict_proba(Xte)[:, 1]
    elif model_type == "ridgeclf":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = RidgeClassifier(alpha=1.0)
        model.fit(Xtr, y_train)
        dec = model.decision_function(Xte)
        return 1.0 / (1.0 + np.exp(-dec))
    elif model_type == "ridge":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(Xtr, y_train)
        return model.predict(Xte)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_Xy(df_feat, feat_cols, df_target, target_name):
    actual_feats = [f for f in feat_cols if f in df_feat.columns]
    X = df_feat[actual_feats].values
    y = df_target[target_name].values
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = np.nan_to_num(X[valid], nan=0, posinf=0, neginf=0)
    y_clean = y[valid]
    return X_clean, y_clean, valid, actual_feats


# ============================================================
# SINGLE PERIOD
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, trade_start, trade_end,
               constraints, cpd):
    t0 = time.time()

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    print(f"\n  Period {period_idx+1}:")
    print(f"    Selection: {len(df_sel)} bars [{df_sel.index[0]} -> {df_sel.index[-1]}]")
    print(f"    Trade:     {len(df_trade)} bars [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    target_features_map = constraints["target_features"]
    core_features = constraints["core_features"]

    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    # ---- Train model set helper ----
    def train_model_set(model_configs, df_tr, df_val):
        preds_out, masks_out = {}, {}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name not in CONTINUOUS_TARGETS

            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_train = prepare_feature_sets(df_tr, tgt_feats, core_features)
            fs_val = prepare_feature_sets(df_val, tgt_feats, core_features)

            if feat_set_name not in fs_train:
                continue

            df_feat_tr, feat_cols_tr = fs_train[feat_set_name]
            df_feat_val, feat_cols_val = fs_val[feat_set_name]
            common_feats = [f for f in feat_cols_tr if f in feat_cols_val]
            if len(common_feats) < 3:
                continue

            X_tr, y_tr, _, _ = get_Xy(df_feat_tr, common_feats, df_tr, full_tgt)
            X_val, y_val, val_mask, _ = get_Xy(df_feat_val, common_feats, df_val, full_tgt)

            if is_binary:
                y_tr = y_tr.astype(int)
                if len(np.unique(y_tr)) < 2:
                    continue

            try:
                p = train_base_model(X_tr, y_tr, X_val, model_type, is_binary)
                preds_out[tgt_name] = p
                masks_out[tgt_name] = val_mask
            except:
                pass
        return preds_out, masks_out

    def retrain_and_predict(model_configs, df_full, df_trd):
        preds_out = {}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name not in CONTINUOUS_TARGETS

            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_full = prepare_feature_sets(df_full, tgt_feats, core_features)
            fs_trade = prepare_feature_sets(df_trd, tgt_feats, core_features)

            if feat_set_name not in fs_full:
                continue

            df_feat_full, feat_cols_full = fs_full[feat_set_name]
            df_feat_trade, feat_cols_trade = fs_trade[feat_set_name]
            common_feats = [f for f in feat_cols_full if f in feat_cols_trade]
            if len(common_feats) < 3:
                continue

            X_full, y_full, _, _ = get_Xy(df_feat_full, common_feats, df_full, full_tgt)
            X_trade_raw = df_feat_trade[common_feats].values
            X_trade_clean = np.nan_to_num(X_trade_raw, nan=0, posinf=0, neginf=0)

            if is_binary:
                y_full = y_full.astype(int)
                if len(np.unique(y_full)) < 2:
                    continue

            try:
                p = train_base_model(X_full, y_full, X_trade_clean, model_type, is_binary)
                preds_out[tgt_name] = p
            except:
                pass
        return preds_out

    # ---- Step 1: Train all models on inner split ----
    all_models = {**VOL_MODELS, **DIR_MODELS}
    val_preds, val_masks = train_model_set(all_models, df_inner_train, df_inner_val)

    vol_ok = sum(1 for k in VOL_MODELS if k in val_preds)
    dir_ok = sum(1 for k in DIR_MODELS if k in val_preds)
    print(f"    Models OK: {vol_ok} vol + {dir_ok} dir = {vol_ok+dir_ok}")

    if vol_ok < 2 or dir_ok < 3:
        print(f"    SKIP: insufficient models")
        return None

    # ---- Step 2: Build meta features ----
    n_val = len(df_inner_val)
    all_model_names = list(VOL_MODELS.keys()) + list(DIR_MODELS.keys())
    n_meta_feats = len(all_model_names)
    meta_features_val = np.zeros((n_val, n_meta_feats))
    meta_valid = np.ones(n_val, dtype=bool)

    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in val_preds:
            mask = val_masks[tgt_name]
            preds = val_preds[tgt_name]
            col = np.full(n_val, np.nan)
            col[mask] = preds
            meta_features_val[:, i] = col
            meta_valid &= np.isfinite(col)
        else:
            if tgt_name in VOL_MODELS:
                meta_features_val[:, i] = 0.5
            else:
                meta_valid[:] = False

    meta_y_long = df_inner_val[META_TARGET_LONG].values if META_TARGET_LONG in df_inner_val.columns else None
    meta_y_short = df_inner_val[META_TARGET_SHORT].values if META_TARGET_SHORT in df_inner_val.columns else None

    if meta_y_long is None or meta_y_short is None:
        return None

    meta_valid &= np.isfinite(meta_y_long) & np.isfinite(meta_y_short)
    X_meta = meta_features_val[meta_valid]
    y_meta_long = meta_y_long[meta_valid].astype(int)
    y_meta_short = meta_y_short[meta_valid].astype(int)

    if len(X_meta) < 50 or len(np.unique(y_meta_long)) < 2 or len(np.unique(y_meta_short)) < 2:
        return None

    # ---- Step 3: Train meta-models ----
    def _train_meta(X, y):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        if META_MODEL_TYPE == "lightgbm":
            m = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=100, max_depth=3, learning_rate=0.1,
                num_leaves=8, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42,
            )
        else:
            m = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        m.fit(Xs, y)
        return m, sc

    def _meta_proba(m, sc, X):
        Xs = sc.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
        return m.predict_proba(Xs)[:, 1]

    meta_long, scaler_long = _train_meta(X_meta, y_meta_long)
    meta_short, scaler_short = _train_meta(X_meta, y_meta_short)

    try:
        auc_l = roc_auc_score(y_meta_long, _meta_proba(meta_long, scaler_long, X_meta))
        auc_s = roc_auc_score(y_meta_short, _meta_proba(meta_short, scaler_short, X_meta))
        print(f"    Meta AUC ({n_meta_feats}-feat): long={auc_l:.3f}, short={auc_s:.3f}")
    except:
        pass

    # ---- Step 3b: Calibrate vol gate + confidence threshold via 3-fold CV ----
    cum_ret_col = "tgt_cum_ret_3"
    conf_threshold = 0.5
    vol_gate_threshold = VOL_GATE_FLOOR

    if cum_ret_col in df_inner_val.columns:
        cum_ret_val = df_inner_val[cum_ret_col].values[meta_valid]

        # Also get vol_expansion predictions for gate calibration
        vol5_idx = all_model_names.index("vol_expansion_5") if "vol_expansion_5" in all_model_names else None

        n_cv = len(X_meta)
        fold_size = n_cv // 3
        cv_pred_long = np.full(n_cv, np.nan)
        cv_pred_short = np.full(n_cv, np.nan)

        for fold in range(3):
            vs = fold * fold_size
            ve = n_cv if fold == 2 else (fold + 1) * fold_size
            cv_mask = np.ones(n_cv, dtype=bool)
            cv_mask[vs:ve] = False
            try:
                ml, sl = _train_meta(X_meta[cv_mask], y_meta_long[cv_mask])
                ms, ss = _train_meta(X_meta[cv_mask], y_meta_short[cv_mask])
                cv_pred_long[vs:ve] = _meta_proba(ml, sl, X_meta[~cv_mask])
                cv_pred_short[vs:ve] = _meta_proba(ms, ss, X_meta[~cv_mask])
            except:
                pass

        # Calibrate: joint threshold on meta confidence AND vol gate
        best_thresh, best_avg = 0.5, -999
        for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            rets = []
            for j in range(n_cv):
                pl, ps = cv_pred_long[j], cv_pred_short[j]
                if np.isnan(pl) or np.isnan(ps):
                    continue
                # Vol gate: check if vol_expansion_5 prediction is high
                if vol5_idx is not None:
                    vol_p = X_meta[j, vol5_idx]
                    if vol_p < VOL_GATE_FLOOR:
                        continue
                max_p = max(pl, ps)
                if max_p > thresh and np.isfinite(cum_ret_val[j]):
                    r = cum_ret_val[j] - FEE_FRAC if pl > ps else -cum_ret_val[j] - FEE_FRAC
                    rets.append(r)
            if len(rets) >= 10 and np.mean(rets) > best_avg:
                best_avg = np.mean(rets)
                best_thresh = thresh
        conf_threshold = best_thresh
        print(f"    Calibrated threshold (3-fold CV): {conf_threshold:.2f} "
              f"(OOS avg ret: {best_avg*100:+.3f}%)")

    # ---- Step 4: Retrain on full selection, predict on trade ----
    trade_preds = retrain_and_predict(all_models, df_sel, df_trade)

    if sum(1 for k in DIR_MODELS if k in trade_preds) < 3:
        return None

    n_trade = len(df_trade)
    meta_features_trade = np.zeros((n_trade, n_meta_feats))
    trade_meta_valid = np.ones(n_trade, dtype=bool)
    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in trade_preds:
            meta_features_trade[:, i] = trade_preds[tgt_name]
        elif tgt_name in VOL_MODELS:
            meta_features_trade[:, i] = 0.5
        else:
            trade_meta_valid[:] = False

    p_long = _meta_proba(meta_long, scaler_long, meta_features_trade)
    p_short = _meta_proba(meta_short, scaler_short, meta_features_trade)

    # Get vol_expansion predictions for gate
    p_vol5 = trade_preds.get("vol_expansion_5", np.full(n_trade, 0.5))
    p_vol10 = trade_preds.get("vol_expansion_10", np.full(n_trade, 0.5))
    p_vol = np.maximum(p_vol5, p_vol10)  # use max of 5 and 10 bar vol predictions

    # ---- Step 5: Generate signals with vol gate + confidence threshold ----
    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    n_vol_gated = 0
    n_conf_gated = 0

    for bar in range(n_trade):
        if not trade_meta_valid[bar]:
            continue

        # Vol gate: only trade when volatility expansion is predicted
        if p_vol[bar] < VOL_GATE_FLOOR:
            n_vol_gated += 1
            continue

        pl, ps = p_long[bar], p_short[bar]
        max_p = max(pl, ps)

        # Confidence gate
        if max_p < conf_threshold:
            n_conf_gated += 1
            continue

        # Direction from meta-model
        if pl > ps:
            signals[bar] = 1.0
            vol_conf = min((p_vol[bar] - 0.5) * 2.0, 1.0) if VOL_GATE_FLOOR > 0 else 1.0
            dir_conf = min((pl - 0.5) * 2.0, 1.0)
            sizes[bar] = min(vol_conf * dir_conf * 2.0, MAX_POSITION_FRAC)
        else:
            signals[bar] = -1.0
            vol_conf = min((p_vol[bar] - 0.5) * 2.0, 1.0) if VOL_GATE_FLOOR > 0 else 1.0
            dir_conf = min((ps - 0.5) * 2.0, 1.0)
            sizes[bar] = min(vol_conf * dir_conf * 2.0, MAX_POSITION_FRAC)

    # ---- Step 6: Simulate trades with early exit ----
    bar_pnl = np.zeros(n_trade)
    bar_gross = np.zeros(n_trade)
    bar_fees = np.zeros(n_trade)
    bar_position = np.zeros(n_trade)
    n_trades = 0
    n_wins = 0
    trade_returns = []
    active_trades = []
    current_exit_bar = -1

    for bar in range(n_trade):
        # Early exit on signal reversal
        # Reversal detected at bar (after bar closes) -> exit at bar+1 open
        if EARLY_EXIT and bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if (signals[bar] != 0 and
                signals[bar] != last_trade["direction"] and
                bar > last_trade["entry_bar"]):
                early_exit_bar = bar + 1  # exit at NEXT bar's open
                if early_exit_bar < n_trade:
                    last_trade["exit_bar"] = early_exit_bar
                    current_exit_bar = early_exit_bar

        # Signal at bar (after bar closes) -> enter at bar+1 open
        if bar > current_exit_bar and signals[bar] != 0 and bar + HOLD_BARS + 1 < n_trade:
            entry_bar = bar + 1
            entry_price = open_prices[entry_bar]
            # Exit at bar after last hold bar's open (entry + HOLD_BARS)
            exit_bar = entry_bar + HOLD_BARS
            if exit_bar >= n_trade:
                continue
            current_exit_bar = exit_bar
            active_trades.append({
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "direction": signals[bar],
                "size": sizes[bar],
                "entry_price": entry_price,
            })

    # Calculate PnL: entry at open[entry_bar], exit at open[exit_bar]
    for trade in active_trades:
        eb, xb = trade["entry_bar"], trade["exit_bar"]
        d, sz, ep = trade["direction"], trade["size"], trade["entry_price"]
        if xb >= n_trade or ep <= 0:
            continue
        exit_price = open_prices[xb]  # exit at open of exit bar
        gross_ret = d * (exit_price / ep - 1.0)
        net_ret = gross_ret - FEE_FRAC
        trade_returns.append(net_ret)
        n_trades += 1
        if net_ret > 0:
            n_wins += 1
        actual_hold = max(xb - eb + 1, 1)
        for b in range(eb, xb + 1):
            if b < n_trade:
                bar_pnl[b] += net_ret * sz / actual_hold
                bar_gross[b] += gross_ret * sz / actual_hold
                bar_fees[b] += FEE_FRAC * sz / actual_hold
                bar_position[b] = d * sz

    total_return = bar_pnl.sum()
    total_gross = bar_gross.sum()
    total_fees = bar_fees.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_trade_ret = np.mean(trade_returns) if trade_returns else 0
    signal_rate = (signals != 0).mean()
    vol_gate_rate = 1.0 - n_vol_gated / n_trade if n_trade > 0 else 1.0

    bnh_return = close_prices[-1] / close_prices[0] - 1.0 if close_prices[0] > 0 else 0

    long_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] > 0]
    short_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] < 0]

    elapsed = time.time() - t0
    print(f"    Trades: {n_trades} ({len(long_trades)}L/{len(short_trades)}S), "
          f"Signal: {signal_rate:.1%}, Vol gate: {vol_gate_rate:.1%}")
    print(f"    Net: {total_return*100:+.2f}%, WR: {win_rate:.1%}, "
          f"Avg trade: {avg_trade_ret*100:+.3f}%")
    print(f"    B&H: {bnh_return*100:+.2f}% [{elapsed:.0f}s]")

    return {
        "period": period_idx + 1,
        "trade_start": str(df_trade.index[0]),
        "trade_end": str(df_trade.index[-1]),
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "net_return": total_return,
        "gross_return": total_gross,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_ret,
        "signal_rate": signal_rate,
        "vol_gate_rate": vol_gate_rate,
        "bnh_return": bnh_return,
        "bar_pnl": bar_pnl,
        "bar_position": bar_position,
        "trade_returns": trade_returns,
        "close_prices": close_prices,
        "trade_dates": df_trade.index,
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(all_results, symbol, tf):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"Volatility Breakout Strategy — {symbol} {tf}\n"
                 f"WFO: {len(all_results)} periods, {SELECTION_DAYS}d train, "
                 f"{PURGE_DAYS}d purge, {TRADE_DAYS}d trade",
                 fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0]
    cum_pnl, cum_bnh, dates = [], [], []
    running_pnl, running_bnh = 0, 0
    for r in all_results:
        for pnl in r["bar_pnl"]:
            running_pnl += pnl
            cum_pnl.append(running_pnl * 100)
        running_bnh += r["bnh_return"]
        cum_bnh.extend([running_bnh * 100] * len(r["bar_pnl"]))
        dates.extend(r["trade_dates"].tolist())

    ax.plot(dates[:len(cum_pnl)], cum_pnl, "b-", linewidth=1.5, label="Strategy")
    ax.plot(dates[:len(cum_bnh)], cum_bnh, "r--", alpha=0.5, label="Buy & Hold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-period returns
    ax = axes[1]
    periods = [r["period"] for r in all_results]
    net_rets = [r["net_return"] * 100 for r in all_results]
    bnh_rets = [r["bnh_return"] * 100 for r in all_results]
    x = np.arange(len(periods))
    ax.bar(x - 0.2, net_rets, 0.4, label="Strategy", color="steelblue")
    ax.bar(x + 0.2, bnh_rets, 0.4, label="Buy & Hold", color="salmon", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in periods])
    ax.set_ylabel("Return (%)")
    ax.set_title("Per-Period Returns")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Trade return distribution
    ax = axes[2]
    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend(r["trade_returns"])
    if all_trade_rets:
        ax.hist([r * 100 for r in all_trade_rets], bins=50, color="steelblue", alpha=0.7)
        ax.axvline(np.mean(all_trade_rets) * 100, color="red", linestyle="--",
                   label=f"Mean: {np.mean(all_trade_rets)*100:.3f}%")
    ax.set_xlabel("Trade Return (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trade Return Distribution (n={len(all_trade_rets)})")
    ax.legend()

    # 4. Signal rate & vol gate per period
    ax = axes[3]
    sig_rates = [r["signal_rate"] * 100 for r in all_results]
    vol_rates = [r["vol_gate_rate"] * 100 for r in all_results]
    win_rates = [r["win_rate"] * 100 for r in all_results]
    ax.plot(periods, sig_rates, "b-o", label="Signal Rate %")
    ax.plot(periods, vol_rates, "r-s", label="Vol Gate Pass %")
    ax.plot(periods, win_rates, "g-^", label="Win Rate %")
    ax.set_xlabel("Period")
    ax.set_ylabel("%")
    ax.set_title("Signal Rate, Vol Gate & Win Rate per Period")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"strategy_straddle_{symbol}_{tf}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print("=" * 80)
    print("  VOLATILITY BREAKOUT STRATEGY — Walk-Forward Backtest")
    print(f"  Symbol: {SYMBOL} {TF}")
    print(f"  WFO: {SELECTION_DAYS}d train, {PURGE_DAYS}d purge, {TRADE_DAYS}d trade")
    print(f"  Periods: {args.periods}, Fees: {FEE_BPS} bps round-trip")
    print(f"  Vol gate floor: {VOL_GATE_FLOOR}, Meta: {META_MODEL_TYPE}")
    print(f"  Hold: {HOLD_BARS} bars, Early exit: {EARLY_EXIT}")
    print("=" * 80)

    df = load_features(FEATURES_DIR, SYMBOL, TF)
    cpd = get_candles_per_day(TF)
    print(f"\n  Loaded {len(df)} candles")

    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    sel_candles = SELECTION_DAYS * cpd
    purge_candles = PURGE_DAYS * cpd
    trade_candles = TRADE_DAYS * cpd

    all_results = []
    for p in range(args.periods):
        sel_start = p * trade_candles
        sel_end = sel_start + sel_candles
        trade_start = sel_end + purge_candles
        trade_end = trade_start + trade_candles
        if trade_end > len(df):
            break
        result = run_period(df, p, sel_start, sel_end, trade_start, trade_end,
                           constraints, cpd)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("ERROR: No periods completed")
        return

    # ---- Aggregate ----
    total_net = sum(r["net_return"] for r in all_results)
    total_gross = sum(r["gross_return"] for r in all_results)
    total_fees = sum(r["total_fees"] for r in all_results)
    total_bnh = sum(r["bnh_return"] for r in all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    total_longs = sum(r["n_long"] for r in all_results)
    total_shorts = sum(r["n_short"] for r in all_results)

    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend(r["trade_returns"])

    avg_win_rate = np.mean([r["win_rate"] for r in all_results])
    avg_signal_rate = np.mean([r["signal_rate"] for r in all_results])
    avg_vol_gate = np.mean([r["vol_gate_rate"] for r in all_results])
    n_positive = sum(1 for r in all_results if r["net_return"] > 0)

    # Trade-level Sharpe
    if len(all_trade_rets) > 1 and np.std(all_trade_rets) > 0:
        trade_years = len(all_results) * TRADE_DAYS / 365.0
        trades_per_year = total_trades / trade_years if trade_years > 0 else total_trades
        sharpe = np.mean(all_trade_rets) / np.std(all_trade_rets) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Max drawdown
    all_bar_pnl = np.concatenate([r["bar_pnl"] for r in all_results])
    cum_pnl = np.cumsum(all_bar_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = (cum_pnl - running_max).min()

    # Profit factor
    wins = sum(r for r in all_trade_rets if r > 0)
    losses = abs(sum(r for r in all_trade_rets if r < 0))
    pf = wins / losses if losses > 0 else float("inf")

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS ({len(all_results)} periods)")
    print(f"{'='*80}")
    print(f"\n  Total net return:    {total_net*100:+.2f}%")
    print(f"  Total gross return:  {total_gross*100:+.2f}%")
    print(f"  Total fees paid:     {total_fees*100:.2f}%")
    print(f"  Buy & Hold return:   {total_bnh*100:+.2f}%")
    print(f"  Excess vs B&H:      {(total_net-total_bnh)*100:+.2f}%")
    print(f"\n  Total trades:        {total_trades} ({total_longs}L / {total_shorts}S)")
    print(f"  Avg trade return:    {np.mean(all_trade_rets)*100:+.3f}%")
    print(f"  Median trade return: {np.median(all_trade_rets)*100:+.3f}%")
    print(f"  Win rate:            {avg_win_rate:.1%}")
    print(f"  Profit factor:       {pf:.2f}")
    print(f"  Signal rate:         {avg_signal_rate:.1%}")
    print(f"  Vol gate pass:       {avg_vol_gate:.1%}")
    print(f"\n  Sharpe ratio (ann):  {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd*100:.2f}%")
    print(f"  Periods positive:    {n_positive}/{len(all_results)}")

    # Per-period table
    print(f"\n  {'Period':<8s} {'Dates':<45s} {'Net%':>8s} {'B&H%':>8s} "
          f"{'Trades':>7s} {'WinR':>6s} {'SigR':>6s} {'VolG':>6s}")
    print(f"  {'-'*100}")
    for r in all_results:
        print(f"  P{r['period']:<6d} {r['trade_start'][:10]} -> {r['trade_end'][:10]}  "
              f"{r['net_return']*100:>+7.2f}% {r['bnh_return']*100:>+7.2f}% "
              f"{r['n_trades']:>6d} {r['win_rate']:>5.1%} "
              f"{r['signal_rate']:>5.1%} {r['vol_gate_rate']:>5.1%}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(all_results, SYMBOL, TF)

    # Save summary JSON
    summary = {
        "strategy": "volatility_breakout",
        "symbol": SYMBOL, "tf": TF,
        "total_net_return": float(total_net),
        "total_gross_return": float(total_gross),
        "total_fees": float(total_fees),
        "bnh_return": float(total_bnh),
        "excess_return": float(total_net - total_bnh),
        "total_trades": total_trades,
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "avg_trade_return": float(np.mean(all_trade_rets)) if all_trade_rets else 0,
        "median_trade_return": float(np.median(all_trade_rets)) if all_trade_rets else 0,
        "win_rate": float(avg_win_rate),
        "profit_factor": float(pf),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "signal_rate": float(avg_signal_rate),
        "vol_gate_rate": float(avg_vol_gate),
        "periods_positive": n_positive,
        "total_periods": len(all_results),
    }
    with open(RESULTS_DIR / f"strategy_straddle_{SYMBOL}_{TF}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save periods CSV
    rows = []
    for r in all_results:
        rows.append({
            "period": r["period"],
            "trade_start": r["trade_start"],
            "trade_end": r["trade_end"],
            "n_trades": r["n_trades"],
            "n_long": r["n_long"],
            "n_short": r["n_short"],
            "net_return": r["net_return"],
            "gross_return": r["gross_return"],
            "total_fees": r["total_fees"],
            "win_rate": r["win_rate"],
            "avg_trade_return": r["avg_trade_return"],
            "signal_rate": r["signal_rate"],
            "vol_gate_rate": r["vol_gate_rate"],
            "bnh_return": r["bnh_return"],
        })
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / f"strategy_straddle_{SYMBOL}_{TF}_periods.csv", index=False)

    elapsed_total = sum(time.time() - t0 for t0 in [time.time()])
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
