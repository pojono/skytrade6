#!/usr/bin/env python3
"""Run exit ML experiments in two phases to avoid OOM:
  Phase 1: Build full dataset with all features, save to parquet
  Phase 2: Load parquet, slice columns per experiment, evaluate
"""
import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered

PARQUET = "exit_ml_experiments_full.parquet"

TARGET_COLS = {"target_drop_remaining", "target_near_bottom_5", "target_near_bottom_10",
               "target_near_bottom_15", "target_bottom_passed"}
META_COLS = {"symbol", "settle_id", "t_ms", "phase"}
ALL_SKIP = TARGET_COLS | META_COLS


def phase1_build():
    """Build dataset with ALL features, save to parquet."""
    from research_exit_ml_experiments import build_dataset
    print("=" * 70)
    print("PHASE 1: BUILD FULL DATASET")
    print("=" * 70)
    df = build_dataset({"ob_depth", "cvd", "sequence", "fr_regime"}, label="all_features")
    if df is None:
        print("FAILED"); return
    df.to_parquet(PARQUET, index=False)
    print(f"\nSaved: {PARQUET} ({len(df)} ticks, {len(df.columns)} cols, "
          f"{Path(PARQUET).stat().st_size/1e6:.1f} MB)")


def evaluate(df, feature_cols, name, do_loso=False):
    """Train LogReg + HGBC on given features."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance

    X = df[feature_cols].values.astype(np.float32)
    y = df["target_near_bottom_10"].values
    symbols = df["symbol"].values

    unique_settle = df["settle_id"].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = df["settle_id"].isin(train_settles).values
    X_tr, X_te = X[train_mask], X[~train_mask]
    y_tr, y_te = y[train_mask], y[~train_mask]

    r = {"name": name, "n_features": len(feature_cols)}

    # LogReg
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("scl", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, max_iter=5000))])
    lr.fit(X_tr, y_tr)
    r["lr_train"] = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    r["lr_test"] = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    r["lr_gap"] = r["lr_train"] - r["lr_test"]
    del lr; gc.collect()

    # HGBC
    hgbc = HistGradientBoostingClassifier(max_iter=300, max_depth=6, min_samples_leaf=30,
                                           learning_rate=0.05, l2_regularization=1.0, random_state=42)
    hgbc.fit(X_tr, y_tr)
    r["hgbc_train"] = roc_auc_score(y_tr, hgbc.predict_proba(X_tr)[:, 1])
    r["hgbc_test"] = roc_auc_score(y_te, hgbc.predict_proba(X_te)[:, 1])
    r["hgbc_gap"] = r["hgbc_train"] - r["hgbc_test"]

    # Top features
    perm = permutation_importance(hgbc, X_te, y_te, n_repeats=3, random_state=42, scoring="roc_auc")
    idx = np.argsort(-perm.importances_mean)
    r["top"] = [(feature_cols[i], perm.importances_mean[i]) for i in idx[:6]]
    del hgbc; gc.collect()

    # LOSO (expensive)
    r["loso"] = float("nan")
    r["bt_loso50"] = float("nan")
    r["bt_fixed10"] = float("nan")
    r["bt_oracle"] = float("nan")

    if do_loso:
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
        n_sym = len(np.unique(symbols))
        print(f"    LOSO CV ({n_sym} symbols)...", end=" ", flush=True)
        logo = LeaveOneGroupOut()
        hgbc2 = HistGradientBoostingClassifier(max_iter=300, max_depth=6, min_samples_leaf=30,
                                                learning_rate=0.05, l2_regularization=1.0, random_state=42)
        yp = cross_val_predict(hgbc2, X, y, cv=logo, groups=symbols, method="predict_proba")[:, 1]
        r["loso"] = roc_auc_score(y, yp)
        print(f"AUC={r['loso']:.4f}")
        del hgbc2; gc.collect()

        # Quick backtest
        bt = df[["settle_id", "t_ms", "price_bps"]].copy()
        bt["p"] = yp
        for sn, th in [("bt_loso50", 0.50)]:
            pnls = []
            for _, sdf in bt.groupby("settle_id"):
                sdf = sdf.sort_values("t_ms")
                ep = sdf.iloc[-1]["price_bps"]
                for _, row in sdf.iterrows():
                    if row["t_ms"] >= 1000 and row["p"] > th:
                        ep = row["price_bps"]; break
                pnls.append(-ep - 20)
            r[sn] = np.mean(pnls)

        # Fixed 10s + oracle
        pnls_f, pnls_o = [], []
        for _, sdf in bt.groupby("settle_id"):
            sdf = sdf.sort_values("t_ms")
            at10 = sdf[sdf["t_ms"] <= 10000]
            pnls_f.append(-(at10.iloc[-1]["price_bps"] if len(at10) else sdf.iloc[-1]["price_bps"]) - 20)
            pnls_o.append(-sdf["price_bps"].min() - 20)
        r["bt_fixed10"] = np.mean(pnls_f)
        r["bt_oracle"] = np.mean(pnls_o)
        del bt, yp; gc.collect()

    return r


def phase2_evaluate():
    """Load parquet, run experiments."""
    print("\n" + "=" * 70)
    print("PHASE 2: EVALUATE EXPERIMENTS")
    print("=" * 70)

    df = pd.read_parquet(PARQUET)
    all_feat = [c for c in df.columns if c not in ALL_SKIP]
    print(f"Loaded: {len(df)} ticks, {len(all_feat)} features")

    # Define column groups
    base = [c for c in all_feat
            if not c.startswith("ob5") and not c.startswith("ob_")
            and not c.startswith("cvd") and c not in (
                "bounce_count", "consecutive_new_lows",
                "price_range_2s", "price_range_5s", "price_std_2s", "price_std_5s",
                "avg_inter_trade_ms", "max_inter_trade_ms", "reversals_2s",
                "fr_x_distance", "fr_x_velocity_1s", "fr_x_time_since_low",
                "fr_x_vol_rate_1s", "fr_x_pct_elapsed", "fr_regime")]
    ob = [c for c in all_feat if c.startswith("ob5") or c.startswith("ob_")]
    cvd = [c for c in all_feat if c.startswith("cvd")]
    seq = [c for c in ["bounce_count", "consecutive_new_lows", "price_range_2s",
           "price_range_5s", "price_std_2s", "price_std_5s",
           "avg_inter_trade_ms", "max_inter_trade_ms", "reversals_2s"] if c in all_feat]
    fr = [c for c in all_feat if c.startswith("fr_x_") or c == "fr_regime"]

    exps = [
        ("Baseline v2",       base,             True),
        ("+ OB depth",        base + ob,        False),
        ("+ CVD",             base + cvd,       False),
        ("+ Sequence",        base + seq,       False),
        ("+ FR regime",       base + fr,        False),
        ("ALL combined",      all_feat,         True),
    ]

    print(f"\nExperiments:")
    for n, cols, loso in exps:
        print(f"  {n:<22s} {len(cols):3d} features {'[+LOSO]' if loso else ''}")

    results = []
    for name, cols, loso in exps:
        print(f"\n{'='*60}")
        print(f"{name} ({len(cols)} features)")
        print(f"{'='*60}")
        t0 = time.time()
        r = evaluate(df, cols, name, do_loso=loso)
        r["desc"] = name
        results.append(r)

        print(f"  LogReg: Train={r['lr_train']:.4f} Test={r['lr_test']:.4f} Gap={r['lr_gap']:+.3f}")
        print(f"  HGBC:   Train={r['hgbc_train']:.4f} Test={r['hgbc_test']:.4f} Gap={r['hgbc_gap']:+.3f}")
        if loso and not np.isnan(r["loso"]):
            print(f"  LOSO:   {r['loso']:.4f}")
            print(f"  BT: LOSO50={r['bt_loso50']:+.1f}  Fixed10={r['bt_fixed10']:+.1f}  Oracle={r['bt_oracle']:+.1f}")
        print(f"  Top: {', '.join(f'{n}({v:.3f})' for n, v in r['top'][:5])}")
        print(f"  [{time.time()-t0:.1f}s]")

    # Summary table
    print(f"\n\n{'='*85}")
    print(f"EXPERIMENT COMPARISON (near_bottom_10)")
    print(f"{'='*85}")
    print(f"{'Experiment':<22s} {'#F':>3s} {'LR Te':>7s} {'HGBC Te':>8s} {'LOSO':>7s} "
          f"{'LRΔ':>7s} {'HGΔ':>7s}")
    print(f"{'-'*22} {'-'*3} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    bl_lr = results[0]["lr_test"]
    bl_hg = results[0]["hgbc_test"]

    for r in results:
        loso_s = f"{r['loso']:.4f}" if not np.isnan(r["loso"]) else "  ---"
        d_lr = r["lr_test"] - bl_lr
        d_hg = r["hgbc_test"] - bl_hg
        print(f"{r['desc']:<22s} {r['n_features']:3d} {r['lr_test']:7.4f} {r['hgbc_test']:8.4f} "
              f"{loso_s:>7s} {d_lr:+7.4f} {d_hg:+7.4f}")

    return results


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "--eval-only":
        phase2_evaluate()
    elif Path(PARQUET).exists():
        print(f"Found {PARQUET}, skipping build. Use --rebuild to force.")
        phase2_evaluate()
    else:
        phase1_build()
        gc.collect()
        phase2_evaluate()
    print(f"\nTotal: {time.time()-t0:.1f}s")
