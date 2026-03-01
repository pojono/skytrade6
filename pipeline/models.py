"""ML models for short exit, long entry decision, and long exit.

Handles training, evaluation, persistence (pickle), and inference.
"""

import copy
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    MODEL_DIR, LONG_ENTRY_MAX_T_S, LONG_EXIT_ML_THRESHOLD,
    LONG_HOLD_FIXED_MS,
)

warnings.filterwarnings("ignore")

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

def save_model(model, feature_cols, name):
    """Save trained model + feature list to disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)
    print(f"    Saved model: {path}")
    return path


def load_model(name):
    """Load model + feature list from disk. Returns (model, feature_cols) or (None, None)."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        return None, None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_cols']


# ═══════════════════════════════════════════════════════════════════════
# SHORT EXIT ML — predict the bottom of the crash
# ═══════════════════════════════════════════════════════════════════════

SHORT_EXIT_SKIP = {
    "symbol", "settle_id", "t_ms", "phase",
    "target_drop_remaining", "target_near_bottom_5",
    "target_near_bottom_10", "target_near_bottom_15",
    "target_bottom_passed",
}


def train_short_exit(tick_df):
    """Train short exit ML models on tick data.

    Returns dict with models, feature_cols, and evaluation metrics.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    print(f"\n  Training short exit ML...")
    t0 = time.time()

    feature_cols = [c for c in tick_df.columns if c not in SHORT_EXIT_SKIP]
    X = tick_df[feature_cols].values
    y = tick_df["target_near_bottom_10"].values
    symbols = tick_df["symbol"].values

    # 70/30 temporal split
    unique_settle = tick_df['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = tick_df['settle_id'].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    results = {}

    # LogReg (production candidate — low overfit)
    lr = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=5000)),
    ])
    lr.fit(X_tr, y_tr)
    auc_tr = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    auc_te = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    print(f"    LogReg: Train AUC={auc_tr:.4f}  Test AUC={auc_te:.4f}  Gap={auc_tr-auc_te:.3f}")
    results['lr_auc_train'] = auc_tr
    results['lr_auc_test'] = auc_te

    # HGBC
    hgbc = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    hgbc.fit(X_tr, y_tr)
    auc_tr_h = roc_auc_score(y_tr, hgbc.predict_proba(X_tr)[:, 1])
    auc_te_h = roc_auc_score(y_te, hgbc.predict_proba(X_te)[:, 1])
    print(f"    HGBC:   Train AUC={auc_tr_h:.4f}  Test AUC={auc_te_h:.4f}  Gap={auc_tr_h-auc_te_h:.3f}")
    results['hgbc_auc_train'] = auc_tr_h
    results['hgbc_auc_test'] = auc_te_h

    # LOSO
    logo = LeaveOneGroupOut()
    try:
        y_pred_loso = cross_val_predict(
            HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=30,
                learning_rate=0.05, l2_regularization=1.0, random_state=42,
            ), X, y, cv=logo, groups=symbols, method='predict_proba'
        )[:, 1]
        auc_loso = roc_auc_score(y, y_pred_loso)
        print(f"    LOSO AUC: {auc_loso:.4f}")
        results['loso_auc'] = auc_loso
    except Exception as e:
        print(f"    LOSO failed: {e}")
        results['loso_auc'] = None

    # Train on ALL data for production
    lr_full = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=5000)),
    ])
    lr_full.fit(X, y)

    hgbc_full = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    hgbc_full.fit(X, y)

    # Save models
    save_model(lr_full, feature_cols, 'short_exit_logreg')
    save_model(hgbc_full, feature_cols, 'short_exit_hgbc')

    results['model_lr'] = lr_full
    results['model_hgbc'] = hgbc_full
    results['feature_cols'] = feature_cols
    results['n_ticks'] = len(tick_df)
    results['n_settle'] = tick_df['settle_id'].nunique()

    print(f"    [{time.time()-t0:.1f}s]")
    return results


# ═══════════════════════════════════════════════════════════════════════
# LONG ENTRY DECISION — rule-based (ML doesn't beat rules with N=105)
# ═══════════════════════════════════════════════════════════════════════

def should_go_long(bottom_t_ms):
    """Rule-based long entry decision.

    Returns True if we should open a long position after closing the short.
    Based on bottom timing — early bottoms (≤15s) have 73% WR vs 41% for late.
    """
    bottom_t_s = bottom_t_ms / 1000.0
    return bottom_t_s <= LONG_ENTRY_MAX_T_S


# ═══════════════════════════════════════════════════════════════════════
# LONG EXIT ML — predict recovery peak
# ═══════════════════════════════════════════════════════════════════════

LONG_EXIT_SKIP = {
    "symbol", "settle_id", "t_ms",
    "target_upside_remaining", "target_near_peak_5",
    "target_near_peak_10", "target_drops_5bps_in_5s",
}


def train_long_exit(tick_df):
    """Train long exit ML models on recovery tick data.

    Returns dict with models, feature_cols, and evaluation metrics.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    print(f"\n  Training long exit ML...")
    t0 = time.time()

    target_col = 'target_near_peak_10'
    feature_cols = [c for c in tick_df.columns if c not in LONG_EXIT_SKIP]

    X = tick_df[feature_cols].values.astype(np.float32)
    y = tick_df[target_col].values.astype(int)
    symbols = tick_df['symbol'].values

    # 70/30 temporal split
    unique_settle = tick_df['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = tick_df['settle_id'].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    results = {}

    # LogReg (zero overfit — production candidate)
    lr = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=5000)),
    ])
    lr.fit(X_tr, y_tr)
    auc_tr = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    auc_te = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    print(f"    LogReg: Train AUC={auc_tr:.4f}  Test AUC={auc_te:.4f}  Gap={auc_tr-auc_te:.3f}")
    results['lr_auc_train'] = auc_tr
    results['lr_auc_test'] = auc_te

    # HGBC
    hgbc = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    hgbc.fit(X_tr, y_tr)
    auc_tr_h = roc_auc_score(y_tr, hgbc.predict_proba(X_tr)[:, 1])
    auc_te_h = roc_auc_score(y_te, hgbc.predict_proba(X_te)[:, 1])
    print(f"    HGBC:   Train AUC={auc_tr_h:.4f}  Test AUC={auc_te_h:.4f}  Gap={auc_tr_h-auc_te_h:.3f}")
    results['hgbc_auc_train'] = auc_tr_h
    results['hgbc_auc_test'] = auc_te_h

    # LOSO
    logo = LeaveOneGroupOut()
    try:
        y_pred_loso = cross_val_predict(
            HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=30,
                learning_rate=0.05, l2_regularization=1.0, random_state=42,
            ), X, y, cv=logo, groups=symbols, method='predict_proba'
        )[:, 1]
        auc_loso = roc_auc_score(y, y_pred_loso)
        print(f"    LOSO AUC: {auc_loso:.4f}")
        results['loso_auc'] = auc_loso
    except Exception as e:
        print(f"    LOSO failed: {e}")
        results['loso_auc'] = None

    # Train on ALL data for production
    lr_full = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=5000)),
    ])
    lr_full.fit(X, y)

    save_model(lr_full, feature_cols, 'long_exit_logreg')

    results['model_lr'] = lr_full
    results['feature_cols'] = feature_cols
    results['n_ticks'] = len(tick_df)
    results['n_settle'] = tick_df['settle_id'].nunique()

    print(f"    [{time.time()-t0:.1f}s]")
    return results


def predict_long_exit(model, feature_cols, recovery_ticks):
    """Given recovery tick features, predict when to exit the long.

    Returns the first tick where p(near_peak) >= threshold,
    or the last tick if threshold never reached.
    """
    if not recovery_ticks:
        return None

    import pandas as pd
    df = pd.DataFrame(recovery_ticks)
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols) * 0.8:
        return None

    # Fill missing columns with 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols].values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]

    # First tick above threshold
    above = np.where(probs >= LONG_EXIT_ML_THRESHOLD)[0]
    if len(above) > 0:
        idx = above[0]
    else:
        idx = len(df) - 1

    return {
        'exit_tick_idx': idx,
        'exit_t_ms': recovery_ticks[idx]['t_ms'],
        'exit_time_since_bottom_ms': recovery_ticks[idx]['time_since_bottom_ms'],
        'exit_recovery_bps': recovery_ticks[idx]['recovery_bps'],
        'exit_prob': float(probs[idx]),
    }
