#!/usr/bin/env python3
"""
Trade Filter Research — Reverse-engineer what happens before bad trades.

Phase 1: Feature engineering on pre-trade conditions
Phase 2: ML classifier (win/loss prediction)
Phase 3: Rule-based filter extraction
Phase 4: Walk-forward backtest with filter
"""

import sys, time, os, json, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

from load_data import load_symbol, list_common_symbols
from features import compute_features
from backtest import compute_composite_signal


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# =============================================================================
# PHASE 1: Build enriched trade dataset with pre-trade features
# =============================================================================

def _process_one_symbol(sym):
    """Load and compute features for one symbol (for parallel execution)."""
    try:
        df = load_symbol(sym)
        if df.empty:
            return sym, None
        feat = compute_features(df)
        sig = compute_composite_signal(feat)
        feat["sig_base"] = sig

        mid = (feat["bb_close"] + feat["bn_close"]) / 2
        mid_ret = mid.pct_change()

        # Rolling volatility at multiple scales (all vectorized, fast)
        feat["rvol_1h"] = mid_ret.rolling(12).std() * np.sqrt(288) * 100
        feat["rvol_4h"] = mid_ret.rolling(48).std() * np.sqrt(288) * 100
        feat["rvol_6h"] = mid_ret.rolling(72).std() * np.sqrt(288) * 100
        feat["rvol_24h"] = mid_ret.rolling(288).std() * np.sqrt(288) * 100
        feat["rvol_ratio_1h_6h"] = feat["rvol_1h"] / feat["rvol_6h"].replace(0, np.nan)
        feat["rvol_ratio_1h_24h"] = feat["rvol_1h"] / feat["rvol_24h"].replace(0, np.nan)

        # Price divergence features
        pdiv = (feat["bb_close"] - feat["bn_close"]) / mid * 10000
        feat["pdiv_bps"] = pdiv
        feat["pdiv_std_1h"] = pdiv.rolling(12).std()
        feat["pdiv_std_6h"] = pdiv.rolling(72).std()
        feat["pdiv_std_24h"] = pdiv.rolling(288).std()
        feat["spread_vol_ratio"] = feat["pdiv_std_1h"] / feat["pdiv_std_6h"].replace(0, np.nan)

        # Momentum
        feat["mid_ret_1h"] = mid.pct_change(12) * 10000
        feat["mid_ret_4h"] = mid.pct_change(48) * 10000
        feat["mid_ret_24h"] = mid.pct_change(288) * 10000

        # Volume features
        bb_vol = feat.get("bb_turnover", feat.get("bb_volume", pd.Series(0, index=feat.index)))
        bn_vol = feat.get("bn_turnover", feat.get("bn_volume", pd.Series(0, index=feat.index)))
        total_vol = bb_vol + bn_vol
        feat["vol_ratio_bb_bn"] = bb_vol / bn_vol.replace(0, np.nan)
        feat["vol_ma12"] = total_vol.rolling(12).mean()
        feat["vol_ma72"] = total_vol.rolling(72).mean()
        feat["vol_ratio_12_72"] = feat["vol_ma12"] / feat["vol_ma72"].replace(0, np.nan)

        # OI features
        bb_oi = feat.get("bb_oi", pd.Series(0, index=feat.index))
        bn_oi = feat.get("bn_oi", pd.Series(0, index=feat.index))
        feat["oi_total"] = bb_oi + bn_oi
        feat["oi_change_1h"] = feat["oi_total"].pct_change(12) * 10000
        feat["oi_change_6h"] = feat["oi_total"].pct_change(72) * 10000

        # Time features
        feat["hour"] = feat.index.hour
        feat["dow"] = feat.index.dayofweek
        feat["is_weekend"] = (feat["dow"] >= 5).astype(int)

        # Cross-exchange correlation (vectorized rolling corr)
        bb_ret = feat["bb_close"].pct_change()
        bn_ret = feat["bn_close"].pct_change()
        feat["cross_corr_12"] = bb_ret.rolling(12).corr(bn_ret)
        feat["cross_corr_72"] = bb_ret.rolling(72).corr(bn_ret)

        # Signal dynamics
        feat["sig_change_3"] = feat["sig_base"].diff(3)
        feat["sig_change_12"] = feat["sig_base"].diff(12)
        feat["sig_abs"] = feat["sig_base"].abs()

        # Trend proxy
        feat["consecutive_up"] = (mid_ret > 0).rolling(12).sum()
        feat["consecutive_down"] = (mid_ret < 0).rolling(12).sum()

        # BTC beta: how much this coin moved vs recent avg (proxy for systemic moves)
        feat["mid_abs_ret_1h"] = mid_ret.rolling(12).apply(lambda x: np.abs(x).mean(), raw=True) * 10000
        feat["mid_abs_ret_6h"] = mid_ret.rolling(72).apply(lambda x: np.abs(x).mean(), raw=True) * 10000
        feat["abs_ret_ratio"] = feat["mid_abs_ret_1h"] / feat["mid_abs_ret_6h"].replace(0, np.nan)

        return sym, feat
    except Exception as e:
        return sym, None


def build_enriched_trades():
    """
    For every trade in production_best_trades.csv, go back to the raw data
    and compute features at the moment of entry.
    """
    trades = pd.read_csv("production_best_trades.csv")
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"])
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"])
    log(f"Loaded {len(trades)} trades from production_best_trades.csv")

    symbols = list(trades["symbol"].unique())
    log(f"Loading data for {len(symbols)} symbols in parallel...")

    symbol_dfs = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_process_one_symbol, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym, feat = fut.result()
            if feat is not None:
                symbol_dfs[sym] = feat
            if done % 20 == 0 or done == len(symbols):
                log(f"  [{done}/{len(symbols)}] {len(symbol_dfs)} loaded ({time.time()-t0:.0f}s)")

    log(f"Loaded {len(symbol_dfs)} symbol dataframes in {time.time()-t0:.0f}s")

    # Now enrich each trade with features at entry time
    feature_cols = [
        "sig_base", "rvol_1h", "rvol_4h", "rvol_6h", "rvol_24h",
        "rvol_ratio_1h_6h", "rvol_ratio_1h_24h",
        "pdiv_bps", "pdiv_std_1h", "pdiv_std_6h", "pdiv_std_24h",
        "spread_vol_ratio",
        "mid_ret_1h", "mid_ret_4h", "mid_ret_24h",
        "vol_ratio_bb_bn", "vol_ratio_12_72",
        "oi_change_1h", "oi_change_6h",
        "hour", "dow", "is_weekend",
        "cross_corr_12", "cross_corr_72",
        "sig_change_3", "sig_change_12", "sig_abs",
        "consecutive_up", "consecutive_down",
        "mid_abs_ret_1h", "mid_abs_ret_6h", "abs_ret_ratio",
    ]

    enriched_rows = []
    missing = 0

    for _, trade in trades.iterrows():
        sym = trade["symbol"]
        if sym not in symbol_dfs:
            missing += 1
            continue

        feat = symbol_dfs[sym]
        entry_idx = trade["entry_i"]

        # Find the closest index position
        if entry_idx >= len(feat) or entry_idx < 0:
            # Try time-based lookup
            entry_time = trade["entry_dt"]
            time_diff = abs(feat.index - entry_time)
            closest = time_diff.argmin()
            if time_diff[closest] > pd.Timedelta(minutes=10):
                missing += 1
                continue
            entry_idx = closest

        row_data = dict(trade)
        row_data["win"] = 1 if trade["net_bps"] > 0 else 0
        row_data["big_win"] = 1 if trade["net_bps"] > 100 else 0
        row_data["big_loss"] = 1 if trade["net_bps"] < -100 else 0

        for col in feature_cols:
            if col in feat.columns:
                val = feat.iloc[entry_idx].get(col, np.nan) if entry_idx < len(feat) else np.nan
                row_data[f"pre_{col}"] = val
            else:
                row_data[f"pre_{col}"] = np.nan

        enriched_rows.append(row_data)

    log(f"Enriched {len(enriched_rows)} trades ({missing} missing)")
    return pd.DataFrame(enriched_rows), feature_cols


# =============================================================================
# PHASE 2: Analyze feature differences between wins and losses
# =============================================================================

def analyze_win_loss(edf, feature_cols):
    """Compare distributions of features between winning and losing trades."""
    log("\n" + "=" * 100)
    log("  PHASE 2: Feature Analysis — Winners vs Losers")
    log("=" * 100)

    pre_cols = [f"pre_{c}" for c in feature_cols]

    wins = edf[edf["win"] == 1]
    losses = edf[edf["win"] == 0]

    log(f"\n  Wins: {len(wins)} ({len(wins)/len(edf)*100:.0f}%)  |  Losses: {len(losses)} ({len(losses)/len(edf)*100:.0f}%)")

    # Compare means with t-test
    from scipy import stats

    results = []
    for col in pre_cols:
        w = wins[col].dropna()
        l = losses[col].dropna()
        if len(w) < 5 or len(l) < 5:
            continue
        t_stat, p_val = stats.ttest_ind(w, l, equal_var=False)
        effect_size = (w.mean() - l.mean()) / max(edf[col].std(), 1e-10)
        results.append({
            "feature": col.replace("pre_", ""),
            "win_mean": w.mean(),
            "loss_mean": l.mean(),
            "diff": w.mean() - l.mean(),
            "effect_size": effect_size,
            "t_stat": t_stat,
            "p_value": p_val,
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    log(f"\n  Top features distinguishing wins vs losses (by p-value):")
    log(f"  {'Feature':>25s} {'Win Mean':>10s} {'Loss Mean':>10s} {'Effect':>8s} {'p-value':>10s}")
    log("  " + "-" * 75)

    for _, r in results_df.head(20).iterrows():
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.1 else ""
        log(f"  {r['feature']:>25s} {r['win_mean']:>10.3f} {r['loss_mean']:>10.3f} "
            f"{r['effect_size']:>+8.3f} {r['p_value']:>10.4f} {sig}")

    # October vs non-October
    log(f"\n\n  --- October vs Non-October ---")
    oct = edf[edf["month"].str.contains("2025-10")]
    non_oct = edf[~edf["month"].str.contains("2025-10")]
    log(f"  October:     {len(oct)} trades, {(oct['win']==1).mean():.0%} WR, avg {oct['net_bps'].mean():+.0f} bps")
    log(f"  Non-October: {len(non_oct)} trades, {(non_oct['win']==1).mean():.0%} WR, avg {non_oct['net_bps'].mean():+.0f} bps")

    log(f"\n  Key feature differences (October vs rest):")
    for col in pre_cols:
        o = oct[col].dropna()
        n = non_oct[col].dropna()
        if len(o) < 3 or len(n) < 3:
            continue
        t, p = stats.ttest_ind(o, n, equal_var=False)
        if p < 0.05:
            log(f"  {col.replace('pre_',''):>25s}: Oct={o.mean():>8.3f} vs Rest={n.mean():>8.3f} (p={p:.4f})")

    return results_df


# =============================================================================
# PHASE 3: ML classifier
# =============================================================================

def train_ml_filter(edf, feature_cols):
    """Train ML models to predict win/loss before trade entry."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    log("\n" + "=" * 100)
    log("  PHASE 3: ML Trade Filter")
    log("=" * 100)

    pre_cols = [f"pre_{c}" for c in feature_cols]
    X = edf[pre_cols].copy()
    y = edf["win"].values

    # Fill NaN with median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Sort by entry time for proper time-series split
    sort_idx = edf["entry_dt"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    edf_sorted = edf.iloc[sort_idx].reset_index(drop=True)

    # Walk-forward cross-validation
    log(f"\n  Dataset: {len(X)} trades, {y.mean():.1%} win rate")
    log(f"  Features: {len(pre_cols)}")

    models = {
        "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                           min_samples_leaf=5, random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                      random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    }

    # Time-series walk-forward: train on first N months, test on next
    months = edf_sorted["month"].values
    unique_months = sorted(edf_sorted["month"].unique())

    log(f"\n  Walk-forward splits: train on months [0..M-1], test on month M")
    log(f"  Months: {unique_months}")

    best_model_name = None
    best_auc = 0
    best_predictions = None

    for model_name, model_template in models.items():
        log(f"\n  --- {model_name} ---")
        all_preds = np.zeros(len(X))
        all_probs = np.zeros(len(X))
        all_mask = np.zeros(len(X), dtype=bool)

        for m_idx in range(2, len(unique_months)):
            test_month = unique_months[m_idx]
            train_months = set(unique_months[:m_idx])

            train_mask = np.array([m in train_months for m in months])
            test_mask = months == test_month

            if train_mask.sum() < 20 or test_mask.sum() < 3:
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Clone model
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train_s, y_train)

            probs = model.predict_proba(X_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)

            all_preds[test_mask] = preds
            all_probs[test_mask] = probs
            all_mask[test_mask] = True

            wr_actual = y_test.mean()
            wr_pred = preds.mean()
            acc = (preds == y_test).mean()

        # Evaluate on all test data
        test_y = y[all_mask]
        test_preds = all_preds[all_mask]
        test_probs = all_probs[all_mask]

        if len(test_y) < 10:
            log(f"  Too few test samples: {len(test_y)}")
            continue

        acc = (test_preds == test_y).mean()
        try:
            auc = roc_auc_score(test_y, test_probs)
        except:
            auc = 0.5

        # Compute trade-level PnL impact
        test_trades = edf_sorted[all_mask].copy()
        test_trades["ml_prob"] = test_probs
        test_trades["ml_take"] = (test_probs >= 0.5).astype(int)

        taken = test_trades[test_trades["ml_take"] == 1]
        skipped = test_trades[test_trades["ml_take"] == 0]

        log(f"  OOS accuracy: {acc:.1%} | AUC: {auc:.3f}")
        log(f"  Trades taken: {len(taken)} ({len(taken)/len(test_trades)*100:.0f}%) | "
            f"Skipped: {len(skipped)} ({len(skipped)/len(test_trades)*100:.0f}%)")

        if len(taken) > 0:
            log(f"  Taken WR: {(taken['net_bps']>0).mean():.1%} | "
                f"Avg: {taken['net_bps'].mean():+.1f} bps | "
                f"Total: {taken['net_bps'].sum():+.0f} bps")
        if len(skipped) > 0:
            log(f"  Skipped WR: {(skipped['net_bps']>0).mean():.1%} | "
                f"Avg: {skipped['net_bps'].mean():+.1f} bps | "
                f"Total: {skipped['net_bps'].sum():+.0f} bps")

        # Monthly breakdown
        log(f"  Monthly (taken only):")
        for month in sorted(taken["month"].unique()):
            mt = taken[taken["month"] == month]
            all_mt = test_trades[test_trades["month"] == month]
            log(f"    {month}: {len(mt)}/{len(all_mt)} trades taken, "
                f"WR={(mt['net_bps']>0).mean():.0%}, "
                f"avg={mt['net_bps'].mean():+.0f} bps, "
                f"total={mt['net_bps'].sum():+.0f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name
            best_predictions = (all_mask, all_probs)

        # Feature importance (for tree models)
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=[c.replace("pre_", "") for c in pre_cols])
            imp = imp.sort_values(ascending=False)
            log(f"  Top 10 features:")
            for feat_name, importance in imp.head(10).items():
                log(f"    {feat_name:>25s}: {importance:.4f}")

    # === Try different probability thresholds ===
    if best_predictions is not None:
        mask, probs = best_predictions
        test_trades = edf_sorted[mask].copy()
        test_trades["ml_prob"] = probs[mask]

        log(f"\n\n  --- Threshold Sweep ({best_model_name}) ---")
        log(f"  {'Threshold':>10s} {'Taken':>7s} {'Skipped':>8s} {'Take WR':>8s} {'Skip WR':>8s} "
            f"{'Take Avg':>9s} {'Skip Avg':>9s} {'Take Total':>11s} {'Skip Total':>11s}")
        log("  " + "-" * 95)

        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            take = test_trades[test_trades["ml_prob"] >= thr]
            skip = test_trades[test_trades["ml_prob"] < thr]
            if len(take) < 3 or len(skip) < 3:
                continue
            take_wr = (take["net_bps"] > 0).mean()
            skip_wr = (skip["net_bps"] > 0).mean()
            log(f"  {thr:>10.2f} {len(take):>7d} {len(skip):>8d} {take_wr:>7.0%} {skip_wr:>7.0%} "
                f"{take['net_bps'].mean():>+9.1f} {skip['net_bps'].mean():>+9.1f} "
                f"{take['net_bps'].sum():>+11.0f} {skip['net_bps'].sum():>+11.0f}")

    return best_model_name, best_predictions


# =============================================================================
# PHASE 4: Rule-based filter extraction
# =============================================================================

def extract_rules(edf, feature_results_df):
    """Find simple rules that filter out losers."""
    log("\n" + "=" * 100)
    log("  PHASE 4: Rule-Based Filter Extraction")
    log("=" * 100)

    # Get top discriminating features
    top_features = feature_results_df.head(10)["feature"].tolist()

    log(f"\n  Testing simple threshold rules on top features...")
    log(f"  Baseline: {len(edf)} trades, {(edf['win']==1).mean():.0%} WR, "
        f"avg {edf['net_bps'].mean():+.0f} bps, total {edf['net_bps'].sum():+.0f} bps")

    best_rules = []

    for feat in top_features:
        col = f"pre_{feat}"
        if col not in edf.columns:
            continue

        vals = edf[col].dropna()
        if len(vals) < 20:
            continue

        # Try percentile-based thresholds
        for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
            thr = np.percentile(vals, pct)

            # Rule: skip if feature < threshold
            take_low = edf[edf[col] >= thr]
            skip_low = edf[edf[col] < thr]

            if len(take_low) >= 20 and len(skip_low) >= 5:
                take_wr = (take_low["net_bps"] > 0).mean()
                skip_wr = (skip_low["net_bps"] > 0).mean()
                take_avg = take_low["net_bps"].mean()
                skip_avg = skip_low["net_bps"].mean()

                # Good rule: taken trades are better than skipped
                if take_avg > skip_avg + 50 and take_wr > skip_wr + 0.05:
                    best_rules.append({
                        "feature": feat,
                        "direction": ">=",
                        "threshold": thr,
                        "percentile": pct,
                        "n_taken": len(take_low),
                        "n_skipped": len(skip_low),
                        "take_wr": take_wr,
                        "skip_wr": skip_wr,
                        "take_avg": take_avg,
                        "skip_avg": skip_avg,
                        "take_total": take_low["net_bps"].sum(),
                        "improvement_bps": take_avg - edf["net_bps"].mean(),
                    })

            # Rule: skip if feature > threshold
            take_high = edf[edf[col] <= thr]
            skip_high = edf[edf[col] > thr]

            if len(take_high) >= 20 and len(skip_high) >= 5:
                take_wr = (take_high["net_bps"] > 0).mean()
                skip_wr = (skip_high["net_bps"] > 0).mean()
                take_avg = take_high["net_bps"].mean()
                skip_avg = skip_high["net_bps"].mean()

                if take_avg > skip_avg + 50 and take_wr > skip_wr + 0.05:
                    best_rules.append({
                        "feature": feat,
                        "direction": "<=",
                        "threshold": thr,
                        "percentile": pct,
                        "n_taken": len(take_high),
                        "n_skipped": len(skip_high),
                        "take_wr": take_wr,
                        "skip_wr": skip_wr,
                        "take_avg": take_avg,
                        "skip_avg": skip_avg,
                        "take_total": take_high["net_bps"].sum(),
                        "improvement_bps": take_avg - edf["net_bps"].mean(),
                    })

    if best_rules:
        rules_df = pd.DataFrame(best_rules).sort_values("improvement_bps", ascending=False)
        log(f"\n  Top 15 Rules (by avg bps improvement over baseline):")
        log(f"  {'Feature':>25s} {'Dir':>3s} {'Thr':>8s} {'Taken':>6s} {'Skip':>6s} "
            f"{'T-WR':>5s} {'S-WR':>5s} {'T-Avg':>8s} {'S-Avg':>8s} {'Improve':>8s}")
        log("  " + "-" * 100)
        for _, r in rules_df.head(15).iterrows():
            log(f"  {r['feature']:>25s} {r['direction']:>3s} {r['threshold']:>8.3f} "
                f"{r['n_taken']:>6.0f} {r['n_skipped']:>6.0f} "
                f"{r['take_wr']:>4.0%} {r['skip_wr']:>4.0%} "
                f"{r['take_avg']:>+8.0f} {r['skip_avg']:>+8.0f} "
                f"{r['improvement_bps']:>+8.0f}")

        return rules_df
    else:
        log("  No significant rules found.")
        return pd.DataFrame()


# =============================================================================
# PHASE 5: Combined filter backtest
# =============================================================================

def backtest_with_filter(edf, rules_df, ml_predictions):
    """Compare PnL with various filter combinations."""
    log("\n" + "=" * 100)
    log("  PHASE 5: Filter Backtest Comparison")
    log("=" * 100)

    BASE = 10000
    edf = edf.sort_values("entry_dt").reset_index(drop=True)

    def evaluate(mask, label):
        taken = edf[mask]
        skipped = edf[~mask]
        n = len(taken)
        wr = (taken["net_bps"] > 0).mean() if n > 0 else 0
        avg = taken["net_bps"].mean() if n > 0 else 0
        total = taken["net_bps"].sum() if n > 0 else 0
        usd = (taken["net_bps"] / 10000 * taken["position_size"] * BASE).sum() if n > 0 else 0

        # Monthly breakdown
        monthly = {}
        for month in sorted(edf["month"].unique()):
            mt = taken[taken["month"] == month]
            if len(mt) > 0:
                monthly[month] = {
                    "n": len(mt),
                    "wr": (mt["net_bps"] > 0).mean(),
                    "pnl": mt["net_bps"].sum(),
                    "usd": (mt["net_bps"] / 10000 * mt["position_size"] * BASE).sum(),
                }
            else:
                monthly[month] = {"n": 0, "wr": 0, "pnl": 0, "usd": 0}

        return {
            "label": label, "n": n, "pct": n / len(edf), "wr": wr,
            "avg_bps": avg, "total_bps": total, "total_usd": usd,
            "monthly": monthly,
        }

    comparisons = []

    # Baseline: no filter
    comparisons.append(evaluate(pd.Series(True, index=edf.index), "No Filter (baseline)"))

    # Top rules
    if len(rules_df) > 0:
        for i, (_, rule) in enumerate(rules_df.head(5).iterrows()):
            col = f"pre_{rule['feature']}"
            if rule["direction"] == ">=":
                mask = edf[col] >= rule["threshold"]
            else:
                mask = edf[col] <= rule["threshold"]
            comparisons.append(evaluate(mask, f"Rule: {rule['feature']} {rule['direction']} {rule['threshold']:.3f}"))

        # Combo: top 2 rules
        if len(rules_df) >= 2:
            r1 = rules_df.iloc[0]
            r2 = rules_df.iloc[1]
            col1 = f"pre_{r1['feature']}"
            col2 = f"pre_{r2['feature']}"
            m1 = edf[col1] >= r1["threshold"] if r1["direction"] == ">=" else edf[col1] <= r1["threshold"]
            m2 = edf[col2] >= r2["threshold"] if r2["direction"] == ">=" else edf[col2] <= r2["threshold"]
            comparisons.append(evaluate(m1 & m2, f"Combo: {r1['feature']} + {r2['feature']}"))

    # ML filter at various thresholds
    if ml_predictions is not None:
        ml_mask, ml_probs = ml_predictions
        edf_ml = edf.copy()
        edf_ml["ml_prob"] = 0.5
        # ml_probs is full-length array, ml_mask is boolean — index properly
        edf_ml.loc[edf_ml.index[ml_mask], "ml_prob"] = ml_probs[ml_mask]

        for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
            mask = edf_ml["ml_prob"] >= thr
            comparisons.append(evaluate(mask, f"ML prob >= {thr:.2f}"))

    # Print comparison
    log(f"\n  {'Filter':>45s} {'N':>5s} {'%':>5s} {'WR':>5s} {'Avg':>8s} {'Total bps':>10s} {'Est USD':>10s}")
    log("  " + "-" * 100)
    for c in comparisons:
        log(f"  {c['label']:>45s} {c['n']:>5d} {c['pct']:>4.0%} {c['wr']:>4.0%} "
            f"{c['avg_bps']:>+8.0f} {c['total_bps']:>+10.0f} {c['total_usd']:>+10.0f}")

    # Monthly comparison for top 3
    log(f"\n  Monthly Comparison (top filters vs baseline):")
    header = f"  {'Month':>10s}"
    for c in comparisons[:4]:
        short = c["label"][:20]
        header += f" {short:>22s}"
    log(header)
    log("  " + "-" * (12 + 23 * min(4, len(comparisons))))

    for month in sorted(edf["month"].unique()):
        row = f"  {month:>10s}"
        for c in comparisons[:4]:
            md = c["monthly"].get(month, {"n": 0, "usd": 0, "wr": 0})
            row += f"  {md['n']:>3d}t {md['wr']:>3.0%} ${md['usd']:>+8,.0f}"
        flag = " ← VOL" if "2025-10" in month else ""
        log(row + flag)

    return comparisons


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    log("=" * 100)
    log("  TRADE FILTER RESEARCH — Claude-Exp-1")
    log("  Reverse-engineering bad trades to build a pre-trade filter")
    log("=" * 100)

    # Phase 1: Build enriched dataset
    log("\n  PHASE 1: Building enriched trade dataset...")
    edf, feature_cols = build_enriched_trades()
    edf.to_csv("enriched_trades.csv", index=False)
    log(f"  Saved {len(edf)} enriched trades to enriched_trades.csv")

    # Phase 2: Analyze features
    results_df = analyze_win_loss(edf, feature_cols)

    # Phase 3: ML filter
    best_model, ml_preds = train_ml_filter(edf, feature_cols)

    # Phase 4: Rule-based filter
    rules_df = extract_rules(edf, results_df)

    # Phase 5: Backtest comparison
    comparisons = backtest_with_filter(edf, rules_df, ml_preds)

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s")
    log(f"  Results saved to enriched_trades.csv")


if __name__ == "__main__":
    main()
