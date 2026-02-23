#!/usr/bin/env python3
"""
Strategy Iteration Runner — Systematic improvements to Directional Momentum

Tests multiple configurations against WFO backtest and compares results.
Each iteration builds on the previous best.

Iterations:
  v0 (baseline): Current 8-feat meta, 3-fold CV threshold
  v1: Expand to 15+ base models using ALL predictable targets with best model/feat
  v2: LightGBM meta-model instead of Logistic (captures nonlinear interactions)
  v3: Early exit on signal reversal (don't hold if signal flips)
  v4: Dynamic hold period (1/3/5 bars based on vol prediction)
  v5: Combined best of all above
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import copy
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ============================================================
# SHARED PARAMETERS
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
MAX_POSITION_FRAC = 1.0

# ============================================================
# ITERATION CONFIGS
# ============================================================

# v0: Current baseline (8 models)
CONFIG_V0 = {
    "name": "v0_baseline",
    "base_models": {
        "breakout_up_3":     {"model": "lightgbm", "features": "core+lags"},
        "breakout_down_3":   {"model": "lightgbm", "features": "core+lags"},
        "profitable_long_1": {"model": "logistic",  "features": "raw+lags"},
        "profitable_short_1":{"model": "logistic",  "features": "raw+lags"},
        "vol_expansion_5":   {"model": "logistic",  "features": "raw"},
        "alpha_1":           {"model": "ridge",     "features": "all_core"},
    },
    "regime_models": {
        "consolidation_3":  {"model": "logistic", "features": "raw+lags"},
        "tail_event_3":     {"model": "logistic", "features": "raw"},
    },
    "meta_model": "logistic",
    "hold_bars": 3,
    "early_exit": False,
    "dynamic_hold": False,
    "meta_target_long": "tgt_profitable_long_3",
    "meta_target_short": "tgt_profitable_short_3",
}

# v1: Expand base models — use ALL strong/moderate targets with best model/feat
CONFIG_V1 = {
    "name": "v1_expanded_models",
    "base_models": {
        # STRONG directional (AUC 0.79-0.86)
        "breakout_up_3":     {"model": "lightgbm",  "features": "core+lags"},
        "breakout_down_3":   {"model": "ridgeclf",  "features": "all_core"},
        "breakout_up_5":     {"model": "lightgbm",  "features": "core+lags"},
        "breakout_down_5":   {"model": "ridgeclf",  "features": "all_core"},
        "breakout_up_10":    {"model": "lightgbm",  "features": "core+lags"},
        "breakout_down_10":  {"model": "ridgeclf",  "features": "all_core"},
        # STRONG volatility (AUC 0.77-0.83)
        "vol_expansion_5":   {"model": "logistic",  "features": "raw"},
        "vol_expansion_10":  {"model": "logistic",  "features": "raw"},
        # MODERATE profitability (AUC 0.55-0.67)
        "profitable_long_1": {"model": "logistic",  "features": "raw+lags"},
        "profitable_short_1":{"model": "logistic",  "features": "raw+lags"},
        "profitable_long_5": {"model": "lightgbm",  "features": "raw+lags"},
        "profitable_short_5":{"model": "lightgbm",  "features": "raw+lags"},
        # STRONG continuous
        "alpha_1":           {"model": "ridge",     "features": "all_core"},
        "relative_ret_1":    {"model": "ridge",     "features": "all_core"},
        # MODERATE risk
        "adverse_selection_1":{"model": "lightgbm", "features": "raw+lags"},
    },
    "regime_models": {
        "consolidation_3":  {"model": "logistic", "features": "raw+lags"},
        "tail_event_3":     {"model": "lightgbm", "features": "raw+lags"},
        "tail_event_5":     {"model": "lightgbm", "features": "all_core"},
        "crash_10":         {"model": "lightgbm", "features": "raw+lags"},
    },
    "meta_model": "logistic",
    "hold_bars": 3,
    "early_exit": False,
    "dynamic_hold": False,
    "meta_target_long": "tgt_profitable_long_3",
    "meta_target_short": "tgt_profitable_short_3",
}

# v2: LightGBM meta-model (nonlinear interactions between base predictions)
CONFIG_V2 = {
    **copy.deepcopy(CONFIG_V1),
    "name": "v2_lgbm_meta",
    "meta_model": "lightgbm",
}

# v3: Early exit on signal reversal
CONFIG_V3 = {
    **copy.deepcopy(CONFIG_V2),
    "name": "v3_early_exit",
    "early_exit": True,
}

# v4: Dynamic hold period
CONFIG_V4 = {
    **copy.deepcopy(CONFIG_V2),
    "name": "v4_dynamic_hold",
    "dynamic_hold": True,
}

# v5: Combined best
CONFIG_V5 = {
    **copy.deepcopy(CONFIG_V2),
    "name": "v5_combined",
    "early_exit": True,
    "dynamic_hold": True,
}


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
        # RidgeClassifier doesn't have predict_proba, use decision_function
        dec = model.decision_function(Xte)
        # Sigmoid to convert to probability
        return 1.0 / (1.0 + np.exp(-dec))
    elif model_type == "ridge":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(Xtr, y_train)
        return model.predict(Xte)
    elif model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
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


def train_meta_model(X, y, model_type="logistic"):
    """Train a meta-model. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if model_type == "logistic":
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(X_scaled, y)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=100, max_depth=3, learning_rate=0.1,
            num_leaves=8, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42,
        )
        model.fit(X_scaled, y)
    else:
        raise ValueError(f"Unknown meta model: {model_type}")
    return model, scaler


def meta_predict_proba(model, scaler, X, model_type="logistic"):
    """Get probability predictions from meta-model."""
    X_scaled = scaler.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
    return model.predict_proba(X_scaled)[:, 1]


# ============================================================
# SINGLE PERIOD
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, trade_start, trade_end,
               constraints, cpd, config):
    t0 = time.time()
    cfg_name = config["name"]
    base_models = config["base_models"]
    regime_models = config["regime_models"]
    meta_model_type = config["meta_model"]
    hold_bars = config["hold_bars"]
    early_exit = config["early_exit"]
    dynamic_hold = config["dynamic_hold"]
    meta_tgt_long = config["meta_target_long"]
    meta_tgt_short = config["meta_target_short"]

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    print(f"\n  P{period_idx+1} [{cfg_name}]:", flush=True)

    target_features_map = constraints["target_features"]
    core_features = constraints["core_features"]

    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    # ---- Train all models on inner split ----
    def train_model_set(model_configs, df_tr, df_val):
        preds_out, masks_out = {}, {}
        continuous_targets = {"alpha_1", "relative_ret_1"}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name not in continuous_targets

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
            except Exception as e:
                pass
        return preds_out, masks_out

    def retrain_and_predict(model_configs, df_full, df_trd):
        preds_out = {}
        continuous_targets = {"alpha_1", "relative_ret_1"}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name not in continuous_targets

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

    # Train base + regime models
    all_models = {**base_models, **regime_models}
    val_all_preds, val_all_masks = train_model_set(all_models, df_inner_train, df_inner_val)

    base_ok = sum(1 for k in base_models if k in val_all_preds)
    if base_ok < 4:
        print(f"    SKIP: only {base_ok} base models")
        return None

    # Build meta features
    n_val = len(df_inner_val)
    all_model_names = list(base_models.keys()) + list(regime_models.keys())
    n_meta_feats = len(all_model_names)
    meta_features_val = np.zeros((n_val, n_meta_feats))
    meta_valid = np.ones(n_val, dtype=bool)

    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in val_all_preds:
            mask = val_all_masks[tgt_name]
            preds = val_all_preds[tgt_name]
            col = np.full(n_val, np.nan)
            col[mask] = preds
            meta_features_val[:, i] = col
            meta_valid &= np.isfinite(col)
        else:
            if tgt_name in regime_models:
                meta_features_val[:, i] = 0.5
            else:
                meta_valid[:] = False

    meta_y_long = df_inner_val[meta_tgt_long].values if meta_tgt_long in df_inner_val.columns else None
    meta_y_short = df_inner_val[meta_tgt_short].values if meta_tgt_short in df_inner_val.columns else None

    if meta_y_long is None or meta_y_short is None:
        return None

    meta_valid &= np.isfinite(meta_y_long) & np.isfinite(meta_y_short)
    X_meta = meta_features_val[meta_valid]
    y_meta_long = meta_y_long[meta_valid].astype(int)
    y_meta_short = meta_y_short[meta_valid].astype(int)

    if len(X_meta) < 50 or len(np.unique(y_meta_long)) < 2 or len(np.unique(y_meta_short)) < 2:
        return None

    # Train meta-models
    meta_long, scaler_meta_l = train_meta_model(X_meta, y_meta_long, meta_model_type)
    meta_short, scaler_meta_s = train_meta_model(X_meta, y_meta_short, meta_model_type)

    # Calibrate threshold via 3-fold CV
    cum_ret_3_col = "tgt_cum_ret_3"
    conf_threshold = 0.5
    if cum_ret_3_col in df_inner_val.columns:
        cum_ret_3_val = df_inner_val[cum_ret_3_col].values[meta_valid]
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
                ml, sl = train_meta_model(X_meta[cv_mask], y_meta_long[cv_mask], meta_model_type)
                ms, ss = train_meta_model(X_meta[cv_mask], y_meta_short[cv_mask], meta_model_type)
                cv_pred_long[vs:ve] = meta_predict_proba(ml, sl, X_meta[~cv_mask], meta_model_type)
                cv_pred_short[vs:ve] = meta_predict_proba(ms, ss, X_meta[~cv_mask], meta_model_type)
            except:
                pass

        best_thresh, best_avg = 0.5, -999
        for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            rets = []
            for j in range(n_cv):
                pl, ps = cv_pred_long[j], cv_pred_short[j]
                if np.isnan(pl) or np.isnan(ps):
                    continue
                if max(pl, ps) > thresh and np.isfinite(cum_ret_3_val[j]):
                    r = cum_ret_3_val[j] - FEE_FRAC if pl > ps else -cum_ret_3_val[j] - FEE_FRAC
                    rets.append(r)
            if len(rets) >= 20 and np.mean(rets) > best_avg:
                best_avg = np.mean(rets)
                best_thresh = thresh
        conf_threshold = best_thresh

    # Retrain on full selection, predict on trade
    trade_all_preds = retrain_and_predict(all_models, df_sel, df_trade)
    if sum(1 for k in base_models if k in trade_all_preds) < 4:
        return None

    n_trade = len(df_trade)
    meta_features_trade = np.zeros((n_trade, n_meta_feats))
    trade_meta_valid = np.ones(n_trade, dtype=bool)
    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in trade_all_preds:
            meta_features_trade[:, i] = trade_all_preds[tgt_name]
        elif tgt_name in regime_models:
            meta_features_trade[:, i] = 0.5
        else:
            trade_meta_valid[:] = False

    p_long = meta_predict_proba(meta_long, scaler_meta_l, meta_features_trade, meta_model_type)
    p_short = meta_predict_proba(meta_short, scaler_meta_s, meta_features_trade, meta_model_type)

    # Vol expansion prediction for dynamic hold
    p_vol = None
    if dynamic_hold and "vol_expansion_5" in trade_all_preds:
        p_vol = trade_all_preds["vol_expansion_5"]

    # Generate signals
    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    hold_periods = np.full(n_trade, hold_bars)
    n_gated_out = 0

    for bar in range(n_trade):
        if not trade_meta_valid[bar]:
            continue
        pl, ps = p_long[bar], p_short[bar]
        max_p = max(pl, ps)
        if max_p < conf_threshold:
            n_gated_out += 1
            continue

        if pl > ps:
            signals[bar] = 1.0
            sizes[bar] = min((pl - 0.5) * 2.0, MAX_POSITION_FRAC)
        else:
            signals[bar] = -1.0
            sizes[bar] = min((ps - 0.5) * 2.0, MAX_POSITION_FRAC)

        # Dynamic hold: longer hold when vol_expansion is predicted
        if dynamic_hold and p_vol is not None:
            if p_vol[bar] > 0.6:
                hold_periods[bar] = 5  # high vol expected → hold longer
            elif p_vol[bar] < 0.4:
                hold_periods[bar] = 1  # low vol → quick scalp
            else:
                hold_periods[bar] = 3  # normal

    # Simulate trades
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
        # Early exit: reversal at bar (after close) -> exit at bar+1 open
        if early_exit and bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if (signals[bar] != 0 and
                signals[bar] != last_trade["direction"] and
                bar > last_trade["entry_bar"]):
                early_exit_bar = bar + 1
                if early_exit_bar < n_trade:
                    last_trade["exit_bar"] = early_exit_bar
                    current_exit_bar = early_exit_bar

        hp = hold_periods[bar]
        # Signal at bar (after close) -> enter at bar+1 open
        if bar > current_exit_bar and signals[bar] != 0 and bar + hp + 1 < n_trade:
            entry_bar = bar + 1
            entry_price = open_prices[entry_bar]
            exit_bar = entry_bar + hp
            if exit_bar >= n_trade:
                continue
            current_exit_bar = exit_bar
            active_trades.append({
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "direction": signals[bar],
                "size": sizes[bar],
                "entry_price": entry_price,
                "hold": hp,
            })

    # PnL: entry at open[entry_bar], exit at open[exit_bar]
    for trade in active_trades:
        eb, xb = trade["entry_bar"], trade["exit_bar"]
        d, sz, ep = trade["direction"], trade["size"], trade["entry_price"]
        hp = trade["hold"]
        if xb >= n_trade or ep <= 0:
            continue
        exit_price = open_prices[xb]  # exit at open of exit bar
        gross_ret = d * (exit_price / ep - 1.0)
        net_ret = gross_ret - FEE_FRAC
        trade_returns.append(net_ret)
        n_trades += 1
        if net_ret > 0:
            n_wins += 1
        for b in range(eb, xb + 1):
            if b < n_trade:
                actual_hold = xb - eb + 1
                bar_pnl[b] += net_ret * sz / max(actual_hold, 1)
                bar_gross[b] += gross_ret * sz / max(actual_hold, 1)
                bar_fees[b] += FEE_FRAC * sz / max(actual_hold, 1)
                bar_position[b] = d * sz

    total_return = bar_pnl.sum()
    total_gross = bar_gross.sum()
    total_fees = bar_fees.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    signal_rate = (signals != 0).mean()
    gate_rate = 1.0 - n_gated_out / n_trade if n_trade > 0 else 1.0

    bnh_return = close_prices[-1] / close_prices[0] - 1.0 if close_prices[0] > 0 else 0

    long_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] > 0]
    short_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] < 0]

    elapsed = time.time() - t0
    print(f"    {n_trades}T ({len(long_trades)}L/{len(short_trades)}S) "
          f"net={total_return*100:+.1f}% WR={win_rate:.0%} "
          f"sig={signal_rate:.0%} [{elapsed:.0f}s]")

    return {
        "period": period_idx + 1,
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "net_return": total_return,
        "gross_return": total_gross,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "signal_rate": signal_rate,
        "gate_rate": gate_rate,
        "bnh_return": bnh_return,
        "bar_pnl": bar_pnl,
        "trade_returns": trade_returns,
        "close_prices": close_prices,
        "trade_dates": df_trade.index,
        "trade_start": str(df_trade.index[0]),
        "trade_end": str(df_trade.index[-1]),
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================
def run_config(df, config, constraints, cpd, n_periods=12):
    name = config["name"]
    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"  Base models: {len(config['base_models'])}, "
          f"Regime: {len(config['regime_models'])}, "
          f"Meta: {config['meta_model']}, "
          f"Hold: {config['hold_bars']}, "
          f"Early exit: {config['early_exit']}, "
          f"Dynamic hold: {config['dynamic_hold']}")
    print(f"{'='*70}")

    n_candles = len(df)
    sel_candles = SELECTION_DAYS * cpd
    purge_candles = PURGE_DAYS * cpd
    trade_candles = TRADE_DAYS * cpd

    all_results = []
    for p in range(n_periods):
        sel_start = p * trade_candles
        sel_end = sel_start + sel_candles
        trade_start = sel_end + purge_candles
        trade_end = trade_start + trade_candles
        if trade_end > n_candles:
            break
        result = run_period(df, p, sel_start, sel_end, trade_start, trade_end,
                           constraints, cpd, config)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return {"name": name, "error": "No periods completed"}

    # Aggregate
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

    summary = {
        "name": name,
        "total_net": total_net,
        "total_gross": total_gross,
        "total_fees": total_fees,
        "bnh": total_bnh,
        "excess": total_net - total_bnh,
        "trades": total_trades,
        "longs": total_longs,
        "shorts": total_shorts,
        "avg_trade_ret": np.mean(all_trade_rets) if all_trade_rets else 0,
        "median_trade_ret": np.median(all_trade_rets) if all_trade_rets else 0,
        "win_rate": avg_win_rate,
        "signal_rate": avg_signal_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": pf,
        "periods_positive": n_positive,
        "total_periods": len(all_results),
        "all_results": all_results,
    }

    print(f"\n  {name}: net={total_net*100:+.1f}% trades={total_trades} "
          f"WR={avg_win_rate:.1%} PF={pf:.2f} Sharpe={sharpe:.2f} "
          f"DD={max_dd*100:.1f}% pos={n_positive}/{len(all_results)}")

    return summary


# ============================================================
# COMPARISON & VISUALIZATION
# ============================================================
def compare_results(all_summaries):
    print(f"\n\n{'='*90}")
    print(f"  ITERATION COMPARISON")
    print(f"{'='*90}")
    print(f"\n  {'Config':<25s} {'Net%':>7s} {'Excess':>8s} {'Trades':>7s} "
          f"{'AvgTrd':>7s} {'WinR':>6s} {'PF':>6s} {'Sharpe':>7s} "
          f"{'MaxDD':>7s} {'Pos':>5s}")
    print(f"  {'-'*95}")

    for s in all_summaries:
        if "error" in s:
            print(f"  {s['name']:<25s} ERROR: {s['error']}")
            continue
        print(f"  {s['name']:<25s} {s['total_net']*100:>+6.1f}% "
              f"{s['excess']*100:>+7.1f}% {s['trades']:>6d} "
              f"{s['avg_trade_ret']*100:>+6.3f}% {s['win_rate']:>5.1%} "
              f"{s['profit_factor']:>5.2f} {s['sharpe']:>6.2f} "
              f"{s['max_dd']*100:>6.1f}% "
              f"{s['periods_positive']}/{s['total_periods']}")

    # Find best by Sharpe
    valid = [s for s in all_summaries if "error" not in s]
    if valid:
        best = max(valid, key=lambda s: s["sharpe"])
        print(f"\n  BEST BY SHARPE: {best['name']} (Sharpe={best['sharpe']:.2f})")
        best_net = max(valid, key=lambda s: s["total_net"])
        print(f"  BEST BY RETURN: {best_net['name']} (net={best_net['total_net']*100:+.1f}%)")
        best_pf = max(valid, key=lambda s: s["profit_factor"])
        print(f"  BEST BY PF:     {best_pf['name']} (PF={best_pf['profit_factor']:.2f})")

    return valid


def plot_comparison(all_summaries, symbol, tf):
    valid = [s for s in all_summaries if "error" not in s and "all_results" in s]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f"Strategy Iteration Comparison — {symbol} {tf}",
                 fontsize=14, fontweight="bold")

    # 1. Equity curves
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for idx, s in enumerate(valid):
        cum = []
        running = 0
        dates = []
        for r in s["all_results"]:
            for pnl in r["bar_pnl"]:
                running += pnl
                cum.append(running * 100)
            dates.extend(r["trade_dates"].tolist())
        ax.plot(dates[:len(cum)], cum, label=s["name"], color=colors[idx], linewidth=1.5)

    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Equity Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Bar chart comparison
    ax = axes[1]
    metrics = ["total_net", "sharpe", "profit_factor", "win_rate"]
    labels = ["Net Return", "Sharpe", "Profit Factor", "Win Rate"]
    x = np.arange(len(metrics))
    width = 0.8 / n

    for idx, s in enumerate(valid):
        vals = [s["total_net"] * 100, s["sharpe"], s["profit_factor"], s["win_rate"] * 100]
        ax.bar(x + idx * width, vals, width, label=s["name"], color=colors[idx])

    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_title("Key Metrics Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"strategy_iterations_{symbol}_{tf}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    print(f"{'='*70}")
    print(f"  STRATEGY ITERATION RUNNER")
    print(f"  Testing {6} configurations on {SYMBOL} {TF}")
    print(f"{'='*70}")

    # Load data
    df = load_features(FEATURES_DIR, SYMBOL, TF)
    cpd = get_candles_per_day(TF)
    print(f"  Loaded {len(df)} candles")

    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    # Check required targets
    required = {"tgt_profitable_long_3", "tgt_profitable_short_3", "tgt_cum_ret_3"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: Missing targets: {missing}")
        sys.exit(1)

    configs = [CONFIG_V0, CONFIG_V1, CONFIG_V2, CONFIG_V3, CONFIG_V4, CONFIG_V5]

    all_summaries = []
    for config in configs:
        summary = run_config(df, config, constraints, cpd, n_periods=12)
        all_summaries.append(summary)

    compare_results(all_summaries)
    plot_comparison(all_summaries, SYMBOL, TF)

    # Save comparison JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = []
    for s in all_summaries:
        d = {k: v for k, v in s.items() if k != "all_results"}
        # Convert numpy types
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        save_data.append(d)

    with open(RESULTS_DIR / f"strategy_iterations_{SYMBOL}_{TF}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
