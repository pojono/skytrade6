#!/usr/bin/env python3
"""
Straddle (Volatility Breakout) Strategy — Iteration Runner

Tests multiple configurations to find the best straddle variant:
  v0: Baseline (vol gate 0.50, 14 models, hold 3, early exit)
  v1: Lower vol gate (0.40) — more trades
  v2: No vol gate (pure meta-model) — maximum trades
  v3: v2 + expanded models (19 total, same as directional v3)
  v4: v3 + hold 2 bars (faster exits for vol capture)
  v5: v3 + simpler sizing (no vol×dir product, just confidence)
  v6: v3 + hold 2 + simpler sizing (combined best)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import copy
import warnings
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

CONTINUOUS_TARGETS = {"alpha_1", "relative_ret_1"}

# ============================================================
# MODEL CONFIGS
# ============================================================
VOL_MODELS_BASE = {
    "vol_expansion_5":   {"model": "logistic",  "features": "raw"},
    "vol_expansion_10":  {"model": "logistic",  "features": "raw"},
    "tail_event_3":      {"model": "lightgbm",  "features": "raw+lags"},
    "tail_event_5":      {"model": "lightgbm",  "features": "all_core"},
    "consolidation_3":   {"model": "logistic",  "features": "raw+lags"},
    "crash_10":          {"model": "lightgbm",  "features": "raw+lags"},
}

DIR_MODELS_BASE = {
    "breakout_up_3":     {"model": "lightgbm",  "features": "core+lags"},
    "breakout_down_3":   {"model": "ridgeclf",  "features": "all_core"},
    "breakout_up_5":     {"model": "lightgbm",  "features": "core+lags"},
    "breakout_down_5":   {"model": "ridgeclf",  "features": "all_core"},
    "profitable_long_1": {"model": "logistic",  "features": "raw+lags"},
    "profitable_short_1":{"model": "logistic",  "features": "raw+lags"},
    "alpha_1":           {"model": "ridge",     "features": "all_core"},
    "adverse_selection_1":{"model": "lightgbm", "features": "raw+lags"},
}

DIR_MODELS_EXPANDED = {
    **DIR_MODELS_BASE,
    "breakout_up_10":    {"model": "lightgbm",  "features": "core+lags"},
    "breakout_down_10":  {"model": "ridgeclf",  "features": "all_core"},
    "profitable_long_5": {"model": "lightgbm",  "features": "raw+lags"},
    "profitable_short_5":{"model": "lightgbm",  "features": "raw+lags"},
    "relative_ret_1":    {"model": "ridge",     "features": "all_core"},
}

# ============================================================
# ITERATION CONFIGS
# ============================================================
CONFIGS = {
    "v0_baseline": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_BASE,
        "meta_type": "lightgbm",
        "hold_bars": 3,
        "early_exit": True,
        "vol_gate_floor": 0.50,
        "sizing": "vol_x_dir",  # vol_conf * dir_conf * 2
    },
    "v1_low_gate": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_BASE,
        "meta_type": "lightgbm",
        "hold_bars": 3,
        "early_exit": True,
        "vol_gate_floor": 0.40,
        "sizing": "vol_x_dir",
    },
    "v2_no_gate": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_BASE,
        "meta_type": "lightgbm",
        "hold_bars": 3,
        "early_exit": True,
        "vol_gate_floor": 0.0,  # no vol gate
        "sizing": "vol_x_dir",
    },
    "v3_expanded": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_EXPANDED,
        "meta_type": "lightgbm",
        "hold_bars": 3,
        "early_exit": True,
        "vol_gate_floor": 0.0,
        "sizing": "vol_x_dir",
    },
    "v4_hold2": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_EXPANDED,
        "meta_type": "lightgbm",
        "hold_bars": 2,
        "early_exit": True,
        "vol_gate_floor": 0.0,
        "sizing": "vol_x_dir",
    },
    "v5_simple_size": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_EXPANDED,
        "meta_type": "lightgbm",
        "hold_bars": 3,
        "early_exit": True,
        "vol_gate_floor": 0.0,
        "sizing": "confidence",  # just (max_p - 0.5) * 2
    },
    "v6_best_combo": {
        "vol_models": VOL_MODELS_BASE,
        "dir_models": DIR_MODELS_EXPANDED,
        "meta_type": "lightgbm",
        "hold_bars": 2,
        "early_exit": True,
        "vol_gate_floor": 0.0,
        "sizing": "confidence",
    },
}


# ============================================================
# HELPERS (same as main script)
# ============================================================
def load_features(features_dir, symbol, tf):
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_candles_per_day(tf):
    return {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}[tf]


def add_lag_features(df, feat_cols, lags=[1, 2, 3]):
    new_cols = {}
    for col in feat_cols:
        if col in df.columns:
            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def prepare_feature_sets(df, target_features, core_features):
    available_raw = [f for f in target_features if f in df.columns]
    available_core = [f for f in core_features if f in df.columns]
    df_lag = add_lag_features(df, available_raw)
    lag_cols = [c for c in df_lag.columns if c not in df.columns and not c.startswith("tgt_")]
    df_core_lag = add_lag_features(df, available_core)
    core_lag_cols = [c for c in df_core_lag.columns if c not in df.columns and not c.startswith("tgt_")]
    return {
        "raw": (df, available_raw),
        "raw+lags": (df_lag, available_raw + lag_cols),
        "all_core": (df, available_core),
        "core+lags": (df_core_lag, available_core + core_lag_cols),
    }


def train_base_model(X_train, y_train, X_test, model_type, is_binary=True):
    if model_type == "lightgbm":
        if is_binary:
            m = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42)
            m.fit(X_train, y_train)
            return m.predict_proba(X_test)[:, 1]
        else:
            m = lgb.LGBMRegressor(
                objective="regression", metric="rmse", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42)
            m.fit(X_train, y_train)
            return m.predict(X_test)
    elif model_type == "logistic":
        sc = StandardScaler()
        return LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(
            sc.fit_transform(X_train), y_train).predict_proba(sc.transform(X_test))[:, 1]
    elif model_type == "ridgeclf":
        sc = StandardScaler()
        m = RidgeClassifier(alpha=1.0)
        m.fit(sc.fit_transform(X_train), y_train)
        dec = m.decision_function(sc.transform(X_test))
        return 1.0 / (1.0 + np.exp(-dec))
    elif model_type == "ridge":
        sc = StandardScaler()
        m = Ridge(alpha=1.0)
        m.fit(sc.fit_transform(X_train), y_train)
        return m.predict(sc.transform(X_test))
    else:
        raise ValueError(f"Unknown: {model_type}")


def get_Xy(df_feat, feat_cols, df_target, target_name):
    actual = [f for f in feat_cols if f in df_feat.columns]
    X = df_feat[actual].values
    y = df_target[target_name].values
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return np.nan_to_num(X[valid], nan=0, posinf=0, neginf=0), y[valid], valid, actual


# ============================================================
# SINGLE PERIOD (parameterized)
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, trade_start, trade_end,
               constraints, cpd, cfg):
    t0 = time.time()
    cfg_name = cfg["name"]
    vol_models = cfg["vol_models"]
    dir_models = cfg["dir_models"]
    meta_type = cfg["meta_type"]
    hold_bars = cfg["hold_bars"]
    early_exit = cfg["early_exit"]
    vol_gate_floor = cfg["vol_gate_floor"]
    sizing_mode = cfg["sizing"]

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    target_features_map = constraints["target_features"]
    core_features = constraints["core_features"]

    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    def train_model_set(model_configs, df_tr, df_val):
        preds_out, masks_out = {}, {}
        for tgt_name, mcfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = mcfg["model"]
            feat_set_name = mcfg["features"]
            is_binary = tgt_name not in CONTINUOUS_TARGETS
            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_train = prepare_feature_sets(df_tr, tgt_feats, core_features)
            fs_val = prepare_feature_sets(df_val, tgt_feats, core_features)
            if feat_set_name not in fs_train:
                continue
            df_feat_tr, feat_cols_tr = fs_train[feat_set_name]
            df_feat_val, feat_cols_val = fs_val[feat_set_name]
            common = [f for f in feat_cols_tr if f in feat_cols_val]
            if len(common) < 3:
                continue
            X_tr, y_tr, _, _ = get_Xy(df_feat_tr, common, df_tr, full_tgt)
            X_val, y_val, val_mask, _ = get_Xy(df_feat_val, common, df_val, full_tgt)
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
        for tgt_name, mcfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = mcfg["model"]
            feat_set_name = mcfg["features"]
            is_binary = tgt_name not in CONTINUOUS_TARGETS
            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_full = prepare_feature_sets(df_full, tgt_feats, core_features)
            fs_trade = prepare_feature_sets(df_trd, tgt_feats, core_features)
            if feat_set_name not in fs_full:
                continue
            df_feat_full, feat_cols_full = fs_full[feat_set_name]
            df_feat_trade, feat_cols_trade = fs_trade[feat_set_name]
            common = [f for f in feat_cols_full if f in feat_cols_trade]
            if len(common) < 3:
                continue
            X_full, y_full, _, _ = get_Xy(df_feat_full, common, df_full, full_tgt)
            X_trade_raw = df_feat_trade[common].values
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

    # Train all models
    all_models = {**vol_models, **dir_models}
    val_preds, val_masks = train_model_set(all_models, df_inner_train, df_inner_val)

    vol_ok = sum(1 for k in vol_models if k in val_preds)
    dir_ok = sum(1 for k in dir_models if k in val_preds)
    if vol_ok < 2 or dir_ok < 3:
        return None

    # Build meta features
    n_val = len(df_inner_val)
    all_model_names = list(vol_models.keys()) + list(dir_models.keys())
    n_meta_feats = len(all_model_names)
    meta_features_val = np.zeros((n_val, n_meta_feats))
    meta_valid = np.ones(n_val, dtype=bool)

    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in val_preds:
            mask = val_masks[tgt_name]
            col = np.full(n_val, np.nan)
            col[mask] = val_preds[tgt_name]
            meta_features_val[:, i] = col
            meta_valid &= np.isfinite(col)
        elif tgt_name in vol_models:
            meta_features_val[:, i] = 0.5
        else:
            meta_valid[:] = False

    meta_y_long = df_inner_val["tgt_profitable_long_3"].values if "tgt_profitable_long_3" in df_inner_val.columns else None
    meta_y_short = df_inner_val["tgt_profitable_short_3"].values if "tgt_profitable_short_3" in df_inner_val.columns else None
    if meta_y_long is None or meta_y_short is None:
        return None

    meta_valid &= np.isfinite(meta_y_long) & np.isfinite(meta_y_short)
    X_meta = meta_features_val[meta_valid]
    y_meta_long = meta_y_long[meta_valid].astype(int)
    y_meta_short = meta_y_short[meta_valid].astype(int)

    if len(X_meta) < 50 or len(np.unique(y_meta_long)) < 2 or len(np.unique(y_meta_short)) < 2:
        return None

    # Train meta-models
    def _train_meta(X, y):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        if meta_type == "lightgbm":
            m = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=100, max_depth=3, learning_rate=0.1,
                num_leaves=8, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42)
        else:
            m = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        m.fit(Xs, y)
        return m, sc

    def _meta_proba(m, sc, X):
        Xs = sc.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
        return m.predict_proba(Xs)[:, 1]

    meta_long, scaler_long = _train_meta(X_meta, y_meta_long)
    meta_short, scaler_short = _train_meta(X_meta, y_meta_short)

    # Calibrate threshold via 3-fold CV
    cum_ret_col = "tgt_cum_ret_3"
    conf_threshold = 0.5

    if cum_ret_col in df_inner_val.columns:
        cum_ret_val = df_inner_val[cum_ret_col].values[meta_valid]
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

        best_thresh, best_avg = 0.5, -999
        for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            rets = []
            for j in range(n_cv):
                pl, ps = cv_pred_long[j], cv_pred_short[j]
                if np.isnan(pl) or np.isnan(ps):
                    continue
                if vol_gate_floor > 0 and vol5_idx is not None:
                    if X_meta[j, vol5_idx] < vol_gate_floor:
                        continue
                max_p = max(pl, ps)
                if max_p > thresh and np.isfinite(cum_ret_val[j]):
                    r = cum_ret_val[j] - FEE_FRAC if pl > ps else -cum_ret_val[j] - FEE_FRAC
                    rets.append(r)
            if len(rets) >= 10 and np.mean(rets) > best_avg:
                best_avg = np.mean(rets)
                best_thresh = thresh
        conf_threshold = best_thresh

    # Retrain on full selection, predict on trade
    trade_preds = retrain_and_predict(all_models, df_sel, df_trade)
    if sum(1 for k in dir_models if k in trade_preds) < 3:
        return None

    n_trade = len(df_trade)
    meta_features_trade = np.zeros((n_trade, n_meta_feats))
    trade_meta_valid = np.ones(n_trade, dtype=bool)
    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in trade_preds:
            meta_features_trade[:, i] = trade_preds[tgt_name]
        elif tgt_name in vol_models:
            meta_features_trade[:, i] = 0.5
        else:
            trade_meta_valid[:] = False

    p_long = _meta_proba(meta_long, scaler_long, meta_features_trade)
    p_short = _meta_proba(meta_short, scaler_short, meta_features_trade)

    p_vol5 = trade_preds.get("vol_expansion_5", np.full(n_trade, 0.5))
    p_vol10 = trade_preds.get("vol_expansion_10", np.full(n_trade, 0.5))
    p_vol = np.maximum(p_vol5, p_vol10)

    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)

    for bar in range(n_trade):
        if not trade_meta_valid[bar]:
            continue
        if vol_gate_floor > 0 and p_vol[bar] < vol_gate_floor:
            continue
        pl, ps = p_long[bar], p_short[bar]
        max_p = max(pl, ps)
        if max_p < conf_threshold:
            continue

        if pl > ps:
            signals[bar] = 1.0
            if sizing_mode == "vol_x_dir":
                vc = min((p_vol[bar] - 0.5) * 2.0, 1.0) if vol_gate_floor > 0 else 1.0
                dc = min((pl - 0.5) * 2.0, 1.0)
                sizes[bar] = min(vc * dc * 2.0, 1.0)
            else:  # confidence
                sizes[bar] = min((pl - 0.5) * 2.0, 1.0)
        else:
            signals[bar] = -1.0
            if sizing_mode == "vol_x_dir":
                vc = min((p_vol[bar] - 0.5) * 2.0, 1.0) if vol_gate_floor > 0 else 1.0
                dc = min((ps - 0.5) * 2.0, 1.0)
                sizes[bar] = min(vc * dc * 2.0, 1.0)
            else:
                sizes[bar] = min((ps - 0.5) * 2.0, 1.0)

    # Simulate trades
    bar_pnl = np.zeros(n_trade)
    trade_returns = []
    active_trades = []
    current_exit_bar = -1
    n_trades = 0
    n_wins = 0

    for bar in range(n_trade):
        if early_exit and bar <= current_exit_bar and len(active_trades) > 0:
            last = active_trades[-1]
            if (signals[bar] != 0 and signals[bar] != last["direction"] and
                bar > last["entry_bar"]):
                last["exit_bar"] = bar
                current_exit_bar = bar

        if bar > current_exit_bar and signals[bar] != 0 and bar + hold_bars < n_trade:
            entry_bar = bar + 1
            if entry_bar < n_trade:
                current_exit_bar = min(entry_bar + hold_bars, n_trade - 1)
                active_trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": current_exit_bar,
                    "direction": signals[bar],
                    "size": sizes[bar],
                    "entry_price": open_prices[entry_bar],
                })

    for trade in active_trades:
        eb, xb = trade["entry_bar"], trade["exit_bar"]
        d, sz, ep = trade["direction"], trade["size"], trade["entry_price"]
        if xb >= n_trade or ep <= 0:
            continue
        exit_price = close_prices[xb]
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

    total_return = bar_pnl.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    signal_rate = (signals != 0).mean()
    bnh = close_prices[-1] / close_prices[0] - 1.0 if close_prices[0] > 0 else 0

    long_count = sum(1 for t in active_trades if t["direction"] > 0)
    short_count = sum(1 for t in active_trades if t["direction"] < 0)

    elapsed = time.time() - t0
    print(f"    {n_trades}T ({long_count}L/{short_count}S) "
          f"net={total_return*100:+.1f}% WR={win_rate:.0%} sig={signal_rate:.0%} "
          f"[{elapsed:.0f}s]")

    return {
        "period": period_idx + 1,
        "n_trades": n_trades,
        "n_long": long_count,
        "n_short": short_count,
        "net_return": total_return,
        "win_rate": win_rate,
        "signal_rate": signal_rate,
        "bnh_return": bnh,
        "bar_pnl": bar_pnl,
        "trade_returns": trade_returns,
        "trade_dates": df_trade.index,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    print("=" * 80)
    print("  STRADDLE STRATEGY — Iteration Runner")
    print(f"  Symbol: {SYMBOL} {TF}")
    print(f"  Configs: {len(CONFIGS)}")
    print("=" * 80)

    df = load_features(FEATURES_DIR, SYMBOL, TF)
    cpd = get_candles_per_day(TF)
    print(f"\n  Loaded {len(df)} candles")

    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    sel_candles = SELECTION_DAYS * cpd
    purge_candles = PURGE_DAYS * cpd
    trade_candles = TRADE_DAYS * cpd

    all_iteration_results = {}

    for cfg_name, cfg_params in CONFIGS.items():
        cfg = {**cfg_params, "name": cfg_name}

        n_vol = len(cfg["vol_models"])
        n_dir = len(cfg["dir_models"])
        print(f"\n{'='*70}")
        print(f"  RUNNING: {cfg_name}")
        print(f"  Vol: {n_vol}, Dir: {n_dir}, Meta: {cfg['meta_type']}, "
              f"Hold: {cfg['hold_bars']}, Exit: {cfg['early_exit']}, "
              f"Gate: {cfg['vol_gate_floor']}, Size: {cfg['sizing']}")
        print(f"{'='*70}")

        results = []
        for p in range(12):
            sel_start = p * trade_candles
            sel_end = sel_start + sel_candles
            trade_start = sel_end + purge_candles
            trade_end = trade_start + trade_candles
            if trade_end > len(df):
                break
            print(f"\n  P{p+1} [{cfg_name}]:")
            r = run_period(df, p, sel_start, sel_end, trade_start, trade_end,
                          constraints, cpd, cfg)
            if r is not None:
                results.append(r)

        if not results:
            print(f"  {cfg_name}: NO RESULTS")
            continue

        # Aggregate
        total_net = sum(r["net_return"] for r in results)
        total_trades = sum(r["n_trades"] for r in results)
        all_rets = []
        for r in results:
            all_rets.extend(r["trade_returns"])

        avg_wr = np.mean([r["win_rate"] for r in results])
        n_pos = sum(1 for r in results if r["net_return"] > 0)

        if len(all_rets) > 1 and np.std(all_rets) > 0:
            trade_years = len(results) * TRADE_DAYS / 365.0
            tpy = total_trades / trade_years if trade_years > 0 else total_trades
            sharpe = np.mean(all_rets) / np.std(all_rets) * np.sqrt(tpy)
        else:
            sharpe = 0

        all_bar_pnl = np.concatenate([r["bar_pnl"] for r in results])
        cum = np.cumsum(all_bar_pnl)
        max_dd = (cum - np.maximum.accumulate(cum)).min()

        wins = sum(r for r in all_rets if r > 0)
        losses = abs(sum(r for r in all_rets if r < 0))
        pf = wins / losses if losses > 0 else float("inf")

        print(f"\n  {cfg_name}: net={total_net*100:+.1f}% trades={total_trades} "
              f"WR={avg_wr:.1%} PF={pf:.2f} Sharpe={sharpe:.2f} "
              f"DD={max_dd*100:.1f}% pos={n_pos}/{len(results)}")

        all_iteration_results[cfg_name] = {
            "net_return": total_net,
            "total_trades": total_trades,
            "avg_trade_return": np.mean(all_rets) if all_rets else 0,
            "win_rate": avg_wr,
            "profit_factor": pf,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "periods_positive": n_pos,
            "total_periods": len(results),
            "signal_rate": np.mean([r["signal_rate"] for r in results]),
            "results": results,
        }

    # ---- Comparison ----
    print(f"\n\n{'='*90}")
    print(f"  ITERATION COMPARISON")
    print(f"{'='*90}")
    print(f"\n  {'Config':<25s} {'Net%':>8s} {'Excess':>8s} {'Trades':>7s} {'AvgTrd':>8s} "
          f"{'WinR':>6s} {'PF':>6s} {'Sharpe':>7s} {'MaxDD':>8s} {'Pos':>5s}")
    print(f"  {'-'*95}")

    bnh_total = sum(r["bnh_return"] for r in list(all_iteration_results.values())[0]["results"])

    best_sharpe_name, best_sharpe_val = "", -999
    best_return_name, best_return_val = "", -999
    best_pf_name, best_pf_val = "", -999

    for name, data in all_iteration_results.items():
        excess = data["net_return"] - bnh_total
        print(f"  {name:<25s} {data['net_return']*100:>+7.1f}% {excess*100:>+7.1f}% "
              f"{data['total_trades']:>6d} {data['avg_trade_return']*100:>+7.3f}% "
              f"{data['win_rate']:>5.1%} {data['profit_factor']:>5.2f} "
              f"{data['sharpe']:>6.2f} {data['max_drawdown']*100:>+7.1f}% "
              f"{data['periods_positive']}/{data['total_periods']}")

        if data["sharpe"] > best_sharpe_val:
            best_sharpe_val = data["sharpe"]
            best_sharpe_name = name
        if data["net_return"] > best_return_val:
            best_return_val = data["net_return"]
            best_return_name = name
        if data["profit_factor"] > best_pf_val:
            best_pf_val = data["profit_factor"]
            best_pf_name = name

    print(f"\n  BEST BY SHARPE: {best_sharpe_name} (Sharpe={best_sharpe_val:.2f})")
    print(f"  BEST BY RETURN: {best_return_name} (net={best_return_val*100:+.1f}%)")
    print(f"  BEST BY PF:     {best_pf_name} (PF={best_pf_val:.2f})")

    # Save comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Straddle Iteration Comparison — {SYMBOL} {TF}", fontsize=14, fontweight="bold")

    for name, data in all_iteration_results.items():
        all_bar_pnl = np.concatenate([r["bar_pnl"] for r in data["results"]])
        cum = np.cumsum(all_bar_pnl) * 100
        all_dates = []
        for r in data["results"]:
            all_dates.extend(r["trade_dates"].tolist())
        ax1.plot(all_dates[:len(cum)], cum, label=name, linewidth=1.2)

    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Equity Curves")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    names = list(all_iteration_results.keys())
    net_rets = [all_iteration_results[n]["net_return"] * 100 for n in names]
    sharpes = [all_iteration_results[n]["sharpe"] for n in names]
    pfs = [all_iteration_results[n]["profit_factor"] for n in names]
    wrs = [all_iteration_results[n]["win_rate"] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.2
    ax2.bar(x - 1.5*w, net_rets, w, label="Net Return %", color="steelblue")
    ax2.bar(x - 0.5*w, sharpes, w, label="Sharpe", color="orange")
    ax2.bar(x + 0.5*w, pfs, w, label="Profit Factor", color="green")
    ax2.bar(x + 1.5*w, wrs, w, label="Win Rate %", color="purple")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_title("Key Metrics Comparison")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"strategy_straddle_iterations_{SYMBOL}_{TF}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Save JSON
    save_data = {}
    for name, data in all_iteration_results.items():
        save_data[name] = {k: v for k, v in data.items() if k != "results"}
        save_data[name]["per_period"] = [
            {"period": r["period"], "net_return": r["net_return"],
             "n_trades": r["n_trades"], "win_rate": r["win_rate"]}
            for r in data["results"]
        ]
    with open(RESULTS_DIR / f"strategy_straddle_iterations_{SYMBOL}_{TF}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
