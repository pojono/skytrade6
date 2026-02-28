#!/usr/bin/env python3
"""Evaluate experiments from saved parquet. No LOSO — just train/test AUC."""
import sys, time, gc, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)

print("Loading parquet...")
df = pd.read_parquet("exit_ml_experiments_full.parquet")

SKIP = {"target_drop_remaining", "target_near_bottom_5", "target_near_bottom_10",
        "target_near_bottom_15", "target_bottom_passed",
        "symbol", "settle_id", "t_ms", "phase"}

all_feat = [c for c in df.columns if c not in SKIP]
print(f"Loaded: {len(df)} ticks, {len(all_feat)} features")

# Column groups
base = [c for c in all_feat
        if not c.startswith("ob5") and not c.startswith("ob_")
        and not c.startswith("cvd") and c not in (
            "bounce_count", "consecutive_new_lows", "price_range_2s", "price_range_5s",
            "price_std_2s", "price_std_5s", "avg_inter_trade_ms", "max_inter_trade_ms",
            "reversals_2s", "fr_x_distance", "fr_x_velocity_1s", "fr_x_time_since_low",
            "fr_x_vol_rate_1s", "fr_x_pct_elapsed", "fr_regime")]
ob = [c for c in all_feat if c.startswith("ob5") or c.startswith("ob_")]
cvd = [c for c in all_feat if c.startswith("cvd")]
seq = [c for c in ["bounce_count", "consecutive_new_lows", "price_range_2s", "price_range_5s",
       "price_std_2s", "price_std_5s", "avg_inter_trade_ms", "max_inter_trade_ms", "reversals_2s"]
       if c in all_feat]
fr = [c for c in all_feat if c.startswith("fr_x_") or c == "fr_regime"]

print(f"  base={len(base)} ob={len(ob)} cvd={len(cvd)} seq={len(seq)} fr={len(fr)}")

exps = [
    ("Baseline v2",      base),
    ("+ OB depth",       base + ob),
    ("+ CVD",            base + cvd),
    ("+ Sequence",       base + seq),
    ("+ FR regime",      base + fr),
    ("ALL combined",     all_feat),
]

# Split
y = df["target_near_bottom_10"].values
unique_settle = df["settle_id"].unique()
n_train = int(len(unique_settle) * 0.7)
train_settles = set(unique_settle[:n_train])
train_mask = df["settle_id"].isin(train_settles).values

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

results = []
for name, cols in exps:
    print(f"\n{'='*60}")
    print(f"{name} ({len(cols)} features)")
    print(f"{'='*60}")
    t0 = time.time()

    X = df[cols].values.astype(np.float32)
    X_tr, X_te = X[train_mask], X[~train_mask]
    y_tr, y_te = y[train_mask], y[~train_mask]

    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, max_iter=5000))])
    lr.fit(X_tr, y_tr)
    lr_tr = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    lr_te = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    del lr

    hg = HistGradientBoostingClassifier(max_iter=300, max_depth=6, min_samples_leaf=30,
                                         learning_rate=0.05, l2_regularization=1.0, random_state=42)
    hg.fit(X_tr, y_tr)
    hg_tr = roc_auc_score(y_tr, hg.predict_proba(X_tr)[:, 1])
    hg_te = roc_auc_score(y_te, hg.predict_proba(X_te)[:, 1])

    perm = permutation_importance(hg, X_te, y_te, n_repeats=3, random_state=42, scoring="roc_auc")
    idx = np.argsort(-perm.importances_mean)
    top = [(cols[i], perm.importances_mean[i]) for i in idx[:6]]
    del hg, X
    gc.collect()

    results.append({"name": name, "nf": len(cols), "lr_tr": lr_tr, "lr_te": lr_te,
                     "hg_tr": hg_tr, "hg_te": hg_te, "top": top})

    print(f"  LogReg: Train={lr_tr:.4f} Test={lr_te:.4f} Gap={lr_tr - lr_te:+.3f}")
    print(f"  HGBC:   Train={hg_tr:.4f} Test={hg_te:.4f} Gap={hg_tr - hg_te:+.3f}")
    print(f"  Top: {', '.join(f'{n}({v:.3f})' for n, v in top[:5])}")
    print(f"  [{time.time() - t0:.1f}s]")

# Summary table
bl_lr = results[0]["lr_te"]
bl_hg = results[0]["hg_te"]
print(f"\n\n{'='*85}")
print(f"COMPARISON TABLE (target: near_bottom_10)")
print(f"{'='*85}")
hdr = f"{'Experiment':<22s} {'#F':>3s} {'LR Test':>8s} {'HGBC Te':>8s} {'LR Gap':>7s} {'HG Gap':>7s} {'LR d':>7s} {'HG d':>7s}"
print(hdr)
print("-" * len(hdr))
for r in results:
    d_lr = r["lr_te"] - bl_lr
    d_hg = r["hg_te"] - bl_hg
    print(f"{r['name']:<22s} {r['nf']:3d} {r['lr_te']:8.4f} {r['hg_te']:8.4f} "
          f"{r['lr_tr'] - r['lr_te']:+7.3f} {r['hg_tr'] - r['hg_te']:+7.3f} "
          f"{d_lr:+7.4f} {d_hg:+7.4f}")
