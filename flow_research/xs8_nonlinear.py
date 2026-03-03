#!/usr/bin/env python3
"""
XS-8b — Nonlinear Model Comparison for Tail Stress Indicator

Loads cached xs8_stress.parquet and compares:
  1. LogisticRegression (baseline)
  2. HistGradientBoostingClassifier (nonlinear, fast)
  3. LogReg with interaction features (breadth×pca, crowd_oi×pca, etc.)
  4. GBT with lag features (stress momentum)

For each model: AUC, quintile uplift, monthly walk-forward stability.
Focuses on best target from XS-8: 12×ATR ≥10% coins (base ~35%).
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs8"
SEED = 42

# Target configs to evaluate (from XS-8 results)
TARGETS = [
    ("tail_frac_8x", 0.20, "8×ATR ≥20%"),
    ("tail_frac_12x", 0.10, "12×ATR ≥10%"),
    ("tail_frac_12x", 0.05, "12×ATR ≥5%"),
]

FEATURE_COLS = ["breadth_extreme", "entropy", "pca_var1", "crowd_fund", "crowd_oi"]


def add_interaction_features(df, base_cols):
    """Add pairwise interactions and squared terms."""
    df = df.copy()
    new_cols = list(base_cols)

    # Key interactions identified from XS-8: crowd_oi is dominant
    pairs = [
        ("crowd_oi", "pca_var1"),
        ("crowd_oi", "breadth_extreme"),
        ("crowd_oi", "crowd_fund"),
        ("pca_var1", "breadth_extreme"),
        ("pca_var1", "entropy"),
        ("crowd_fund", "breadth_extreme"),
    ]
    for a, b in pairs:
        col = f"{a}_x_{b}"
        df[col] = df[a] * df[b]
        new_cols.append(col)

    # Squared terms for top features
    for c in ["crowd_oi", "pca_var1", "crowd_fund"]:
        col = f"{c}_sq"
        df[col] = df[c] ** 2
        new_cols.append(col)

    return df, new_cols


def add_lag_features(df, base_cols, lags=[12, 36, 72]):
    """Add lagged values and momentum (change from lag)."""
    df = df.copy()
    new_cols = list(base_cols)

    for c in ["crowd_oi", "pca_var1", "breadth_extreme"]:
        for lag in lags:
            lag_col = f"{c}_lag{lag}"
            df[lag_col] = df[c].shift(lag)
            new_cols.append(lag_col)

            mom_col = f"{c}_mom{lag}"
            df[mom_col] = df[c] - df[c].shift(lag)
            new_cols.append(mom_col)

    # Rolling stats (1h = 12 bars, 6h = 72 bars)
    for c in ["crowd_oi", "pca_var1"]:
        for win in [12, 72]:
            mean_col = f"{c}_mean{win}"
            std_col = f"{c}_std{win}"
            df[mean_col] = df[c].rolling(win, min_periods=win // 2).mean()
            df[std_col] = df[c].rolling(win, min_periods=win // 2).std()
            new_cols.append(mean_col)
            new_cols.append(std_col)

    return df, new_cols


def evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    """Fit model, return AUC and quintile stats."""
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test) if 0 < y_test.sum() < len(y_test) else np.nan

    # Quintile analysis
    df_t = pd.DataFrame({"p": p_test, "y": y_test})
    try:
        df_t["Q"] = pd.qcut(df_t["p"], 5, labels=["Q1","Q2","Q3","Q4","Q5"],
                             duplicates="drop")
    except ValueError:
        return {"model": model_name, "auc_oos": auc, "q5_q1": np.nan}

    base = y_test.mean()
    q_stats = {}
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        qd = df_t[df_t["Q"] == q]
        q_stats[q] = qd["y"].mean() if len(qd) > 0 else 0

    q5_q1 = q_stats["Q5"] / q_stats["Q1"] if q_stats["Q1"] > 0 else np.inf

    return {
        "model": model_name, "auc_oos": auc, "q5_q1": q5_q1,
        "q1_rate": q_stats["Q1"], "q5_rate": q_stats["Q5"],
        "base_rate": base,
    }


def walk_forward_monthly(df, feature_cols, target_col, model_factory, model_name):
    """Monthly expanding-window walk-forward evaluation."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["ts"]).dt.to_period("M")
    months = sorted(df["month"].unique())

    results = []
    for mi in range(2, len(months)):
        train_mask = df["month"].isin(months[:mi])
        test_mask = df["month"] == months[mi]

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask, feature_cols].values
        y_te = df.loc[test_mask, target_col].values

        if y_tr.sum() < 10 or y_te.sum() < 3 or len(y_te) < 50 or y_te.sum() == len(y_te):
            continue

        try:
            m = model_factory()
            m.fit(X_tr, y_tr)
            p_te = m.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, p_te)
            results.append({"month": str(months[mi]), "auc": auc,
                           "n": len(y_te), "pos": int(y_te.sum())})
        except Exception:
            pass

    return results


def main():
    t0 = time.time()

    print("=" * 80)
    print("XS-8b — NONLINEAR MODEL COMPARISON")
    print("=" * 80, flush=True)

    # Load cached stress features
    parquet_path = OUTPUT_DIR / "xs8_stress.parquet"
    print(f"\nLoading {parquet_path}...")
    stress_df = pd.read_parquet(parquet_path)
    print(f"  {len(stress_df):,} rows, columns: {list(stress_df.columns)}")

    # Build targets
    for frac_col, cutoff, label in TARGETS:
        if frac_col not in stress_df.columns:
            print(f"  ✗ {frac_col} not found, skipping {label}")
            continue
        target_col = f"target_{label.replace(' ', '_').replace('×', 'x').replace('≥', 'ge')}"
        stress_df[target_col] = (stress_df[frac_col] >= cutoff).astype(float)
        stress_df.loc[stress_df[frac_col].isna(), target_col] = np.nan

    # Focus on best target: 12×ATR ≥10%
    print(f"\n{'─'*70}")
    print("PHASE 1: Model comparison on 12×ATR ≥10% target")
    print(f"{'─'*70}", flush=True)

    frac_col = "tail_frac_12x"
    cutoff = 0.10
    target_col = "target_12xATR_ge10pct"
    stress_df[target_col] = (stress_df[frac_col] >= cutoff).astype(float)
    stress_df.loc[stress_df[frac_col].isna(), target_col] = np.nan

    # Prepare base data
    df = stress_df.dropna(subset=FEATURE_COLS + [target_col]).copy()
    df["pca_var1"] = df["pca_var1"].fillna(df["pca_var1"].median())
    n = len(df)
    n_pos = int(df[target_col].sum())
    print(f"  N={n:,}, positives={n_pos:,} ({n_pos/n:.1%})")

    # Walk-forward split: train first 60%, test last 40%
    split = int(n * 0.6)

    # ── Model 1: Logistic Regression (baseline) ──
    print(f"\n  ── M1: LogisticRegression (baseline, 5 features) ──")
    X_train = df[FEATURE_COLS].values[:split]
    X_test = df[FEATURE_COLS].values[split:]
    y_train = df[target_col].values[:split]
    y_test = df[target_col].values[split:]

    r1 = evaluate_model(X_train, y_train, X_test, y_test,
                        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
                        "LogReg_5feat")
    print(f"  AUC={r1['auc_oos']:.4f}, Q5/Q1={r1['q5_q1']:.2f}×, "
          f"Q1={r1['q1_rate']:.1%}, Q5={r1['q5_rate']:.1%}")

    # ── Model 2: HistGBT (nonlinear, 5 features) ──
    print(f"\n  ── M2: HistGradientBoosting (5 features) ──")
    r2 = evaluate_model(X_train, y_train, X_test, y_test,
                        HistGradientBoostingClassifier(
                            max_iter=200, max_depth=4, learning_rate=0.05,
                            min_samples_leaf=100, random_state=SEED),
                        "HGBT_5feat")
    print(f"  AUC={r2['auc_oos']:.4f}, Q5/Q1={r2['q5_q1']:.2f}×, "
          f"Q1={r2['q1_rate']:.1%}, Q5={r2['q5_rate']:.1%}")

    # ── Model 3: LogReg with interaction features ──
    print(f"\n  ── M3: LogReg + interactions (14 features) ──")
    df_int, int_cols = add_interaction_features(df, FEATURE_COLS)
    df_int = df_int.dropna(subset=int_cols + [target_col])
    split_int = int(len(df_int) * 0.6)

    scaler = StandardScaler()
    X_train_int = scaler.fit_transform(df_int[int_cols].values[:split_int])
    X_test_int = scaler.transform(df_int[int_cols].values[split_int:])
    y_train_int = df_int[target_col].values[:split_int]
    y_test_int = df_int[target_col].values[split_int:]

    r3 = evaluate_model(X_train_int, y_train_int, X_test_int, y_test_int,
                        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
                        "LogReg_interact")
    print(f"  AUC={r3['auc_oos']:.4f}, Q5/Q1={r3['q5_q1']:.2f}×, "
          f"Q1={r3['q1_rate']:.1%}, Q5={r3['q5_rate']:.1%}")

    # ── Model 4: HGBT with interaction features ──
    print(f"\n  ── M4: HGBT + interactions (14 features) ──")
    r4 = evaluate_model(X_train_int, y_train_int, X_test_int, y_test_int,
                        HistGradientBoostingClassifier(
                            max_iter=200, max_depth=4, learning_rate=0.05,
                            min_samples_leaf=100, random_state=SEED),
                        "HGBT_interact")
    print(f"  AUC={r4['auc_oos']:.4f}, Q5/Q1={r4['q5_q1']:.2f}×, "
          f"Q1={r4['q1_rate']:.1%}, Q5={r4['q5_rate']:.1%}")

    # ── Model 5: HGBT with lag + momentum features ──
    print(f"\n  ── M5: HGBT + lags + momentum (rich features) ──")
    df_lag, lag_cols = add_lag_features(df, FEATURE_COLS)
    df_lag_int, all_cols = add_interaction_features(df_lag, lag_cols)
    df_rich = df_lag_int.dropna(subset=all_cols + [target_col])
    split_rich = int(len(df_rich) * 0.6)

    X_train_rich = df_rich[all_cols].values[:split_rich]
    X_test_rich = df_rich[all_cols].values[split_rich:]
    y_train_rich = df_rich[target_col].values[:split_rich]
    y_test_rich = df_rich[target_col].values[split_rich:]

    print(f"  Features: {len(all_cols)}, N_train={split_rich:,}, N_test={len(df_rich)-split_rich:,}")

    r5 = evaluate_model(X_train_rich, y_train_rich, X_test_rich, y_test_rich,
                        HistGradientBoostingClassifier(
                            max_iter=300, max_depth=5, learning_rate=0.03,
                            min_samples_leaf=200, l2_regularization=1.0,
                            random_state=SEED),
                        "HGBT_rich")
    print(f"  AUC={r5['auc_oos']:.4f}, Q5/Q1={r5['q5_q1']:.2f}×, "
          f"Q1={r5['q1_rate']:.1%}, Q5={r5['q5_rate']:.1%}")

    # ── Model 6: Tuned HGBT (deeper, more trees) ──
    print(f"\n  ── M6: HGBT tuned (deeper, 5 base features) ──")
    r6 = evaluate_model(X_train, y_train, X_test, y_test,
                        HistGradientBoostingClassifier(
                            max_iter=500, max_depth=6, learning_rate=0.02,
                            min_samples_leaf=50, l2_regularization=0.5,
                            random_state=SEED),
                        "HGBT_tuned")
    print(f"  AUC={r6['auc_oos']:.4f}, Q5/Q1={r6['q5_q1']:.2f}×, "
          f"Q1={r6['q1_rate']:.1%}, Q5={r6['q5_rate']:.1%}")

    # ── Summary Table ──
    all_results = [r1, r2, r3, r4, r5, r6]
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY (12×ATR ≥10%, OOS)")
    print(f"{'='*80}")
    print(f"  {'Model':25s}  {'AUC':>7s}  {'Q5/Q1':>7s}  {'Q1':>7s}  {'Q5':>7s}  {'Base':>7s}")
    for r in all_results:
        print(f"  {r['model']:25s}  {r['auc_oos']:>7.4f}  {r['q5_q1']:>6.2f}×  "
              f"{r['q1_rate']:>6.1%}  {r['q5_rate']:>6.1%}  {r['base_rate']:>6.1%}")

    best = max(all_results, key=lambda r: r.get("auc_oos", 0))
    print(f"\n  Best: {best['model']} — AUC={best['auc_oos']:.4f}, Q5/Q1={best['q5_q1']:.2f}×")

    # ── Phase 2: Monthly Walk-Forward for top models ──
    print(f"\n{'─'*70}")
    print("PHASE 2: Monthly walk-forward stability (top models)")
    print(f"{'─'*70}", flush=True)

    model_configs = [
        ("LogReg_5feat", FEATURE_COLS, df,
         lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ("HGBT_5feat", FEATURE_COLS, df,
         lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                     learning_rate=0.05, min_samples_leaf=100, random_state=SEED)),
        ("HGBT_tuned", FEATURE_COLS, df,
         lambda: HistGradientBoostingClassifier(max_iter=500, max_depth=6,
                     learning_rate=0.02, min_samples_leaf=50,
                     l2_regularization=0.5, random_state=SEED)),
    ]

    # Also add rich features model if it had good AUC
    if r5["auc_oos"] > r1["auc_oos"]:
        model_configs.append(
            ("HGBT_rich", all_cols, df_rich,
             lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=5,
                         learning_rate=0.03, min_samples_leaf=200,
                         l2_regularization=1.0, random_state=SEED))
        )

    wf_summary = {}
    for mname, fcols, mdf, mfactory in model_configs:
        print(f"\n  {mname}:")
        wf = walk_forward_monthly(mdf, fcols, target_col, mfactory, mname)
        if wf:
            aucs = [w["auc"] for w in wf]
            wf_summary[mname] = aucs
            for w in wf:
                print(f"    → {w['month']}: AUC={w['auc']:.4f} "
                      f"(N={w['n']:,}, pos={w['pos']:,})")
            print(f"    Mean={np.mean(aucs):.4f}, Std={np.std(aucs):.4f}, "
                  f"Min={np.min(aucs):.4f}, Max={np.max(aucs):.4f}")

    # ── Phase 3: Multi-target comparison with best model ──
    print(f"\n{'─'*70}")
    print("PHASE 3: Best model across multiple targets")
    print(f"{'─'*70}", flush=True)

    best_model_name = best["model"]
    print(f"  Using: {best_model_name}")

    multi_results = []
    for frac_col, cutoff, label in TARGETS:
        tc = f"mt_{frac_col}_{int(cutoff*100)}"
        stress_df[tc] = (stress_df[frac_col] >= cutoff).astype(float)
        stress_df.loc[stress_df[frac_col].isna(), tc] = np.nan

        mdf = stress_df.dropna(subset=FEATURE_COLS + [tc]).copy()
        mdf["pca_var1"] = mdf["pca_var1"].fillna(mdf["pca_var1"].median())

        rate = mdf[tc].mean()
        if rate > 0.95 or rate < 0.02:
            print(f"  {label}: rate={rate:.1%} (skipped)")
            continue

        sp = int(len(mdf) * 0.6)
        Xtr = mdf[FEATURE_COLS].values[:sp]
        Xte = mdf[FEATURE_COLS].values[sp:]
        ytr = mdf[tc].values[:sp]
        yte = mdf[tc].values[sp:]

        # LogReg
        r_lr = evaluate_model(Xtr, ytr, Xte, yte,
                              LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
                              "LogReg")
        # HGBT tuned
        r_gb = evaluate_model(Xtr, ytr, Xte, yte,
                              HistGradientBoostingClassifier(
                                  max_iter=500, max_depth=6, learning_rate=0.02,
                                  min_samples_leaf=50, l2_regularization=0.5,
                                  random_state=SEED),
                              "HGBT")

        multi_results.append({
            "target": label, "base": rate,
            "lr_auc": r_lr["auc_oos"], "lr_q5q1": r_lr["q5_q1"],
            "gb_auc": r_gb["auc_oos"], "gb_q5q1": r_gb["q5_q1"],
            "auc_gain": r_gb["auc_oos"] - r_lr["auc_oos"],
        })
        print(f"  {label:20s}  base={rate:.1%}  "
              f"LR: AUC={r_lr['auc_oos']:.4f} Q5/Q1={r_lr['q5_q1']:.2f}×  "
              f"GBT: AUC={r_gb['auc_oos']:.4f} Q5/Q1={r_gb['q5_q1']:.2f}×  "
              f"Δ={r_gb['auc_oos']-r_lr['auc_oos']:+.4f}")

    # ── Feature importance from best GBT ──
    print(f"\n{'─'*70}")
    print("PHASE 4: GBT Feature Importance")
    print(f"{'─'*70}", flush=True)

    gbt = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.02,
        min_samples_leaf=50, l2_regularization=0.5, random_state=SEED)
    gbt.fit(X_train, y_train)

    # Permutation importance (more reliable than built-in)
    base_auc = roc_auc_score(y_test, gbt.predict_proba(X_test)[:, 1])
    print(f"  Base AUC: {base_auc:.4f}")
    print(f"\n  Permutation importance (drop in AUC when feature shuffled):")
    rng = np.random.default_rng(SEED)
    for i, fname in enumerate(FEATURE_COLS):
        drops = []
        for _ in range(10):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_auc = roc_auc_score(y_test, gbt.predict_proba(X_perm)[:, 1])
            drops.append(base_auc - perm_auc)
        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        print(f"    {fname:20s}: ΔAUC = {mean_drop:+.4f} ± {std_drop:.4f}")

    # Save everything
    if multi_results:
        pd.DataFrame(multi_results).to_csv(OUTPUT_DIR / "xs8_nonlinear_summary.csv", index=False)

    # ── Final Verdict ──
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")

    if multi_results:
        avg_gain = np.mean([r["auc_gain"] for r in multi_results])
        print(f"\n  Average AUC gain from GBT over LogReg: {avg_gain:+.4f}")

        if avg_gain > 0.02:
            print(f"  → Nonlinear models provide MEANINGFUL improvement ✅")
        elif avg_gain > 0.005:
            print(f"  → Nonlinear models provide MARGINAL improvement ⚠️")
        else:
            print(f"  → Nonlinear models provide NO improvement ❌")
            print(f"    The signal is inherently linear (crowd_oi dominance)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
