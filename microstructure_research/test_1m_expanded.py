#!/usr/bin/env python3
"""
1m predictability test with EXPANDED features:
  - Longer rolling horizons: 4h, 12h, 24h
  - Lagged features: what was the state 1h, 4h, 12h ago?
  - Rate-of-change features: how are things changing?
  - Cross-feature interactions

Then re-run predictability scan + P&L test.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = Path("./parquet/SOLUSDT/1m_cache")
SYMBOL = "SOLUSDT"
START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0


# ============================================================
# DATA LOADING
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


# ============================================================
# EXPANDED FEATURE ENGINEERING
# ============================================================
def add_expanded_features(df):
    """
    Build a rich feature set with:
    1. Multi-horizon rolling stats (5m to 24h)
    2. Lagged snapshots (1h, 4h, 12h ago)
    3. Rate-of-change (delta between horizons)
    4. Cross-feature ratios
    """
    t0 = time.time()
    bph = 60  # bars per hour

    # ---- 1. MULTI-HORIZON ROLLING FEATURES ----
    print("  [1/4] Multi-horizon rolling features...", flush=True)

    # Volatility at multiple scales
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"),
                     (720, "12h"), (1440, "24h")]:
        df[f"rvol_{label}"] = df["returns"].rolling(w, min_periods=w//2).std()

    # Volume stats at multiple scales
    for w, label in [(15, "15m"), (60, "1h"), (240, "4h"), (1440, "24h")]:
        vol_roll = df["volume"].rolling(w, min_periods=w//2)
        df[f"vol_mean_{label}"] = vol_roll.mean()
        df[f"vol_zscore_{label}"] = (df["volume"] - vol_roll.mean()) / vol_roll.std().clip(lower=1e-10)

    # Arrival rate at multiple scales
    for w, label in [(15, "15m"), (60, "1h"), (240, "4h")]:
        rate_roll = df["arrival_rate"].rolling(w, min_periods=w//2)
        df[f"rate_zscore_{label}"] = (df["arrival_rate"] - rate_roll.mean()) / rate_roll.std().clip(lower=1e-10)

    # Price momentum at multiple scales
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"),
                     (720, "12h"), (1440, "24h")]:
        df[f"mom_{label}"] = df["close"].pct_change(w)

    # Price z-score (mean reversion signal) at multiple scales
    for w, label in [(60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")]:
        ma = df["close"].rolling(w, min_periods=w//2).mean()
        std = df["close"].rolling(w, min_periods=w//2).std()
        df[f"price_zscore_{label}"] = (df["close"] - ma) / std.clip(lower=1e-10)

    # Range z-score at multiple scales
    for w, label in [(60, "1h"), (240, "4h"), (1440, "24h")]:
        rng_roll = df["price_range"].rolling(w, min_periods=w//2)
        df[f"range_zscore_{label}"] = (df["price_range"] - rng_roll.mean()) / rng_roll.std().clip(lower=1e-10)

    # Cumulative order flow imbalance at multiple scales
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"),
                     (720, "12h")]:
        df[f"cum_imbalance_{label}"] = df["vol_imbalance"].rolling(w, min_periods=w//2).sum()

    # Efficiency ratio at multiple scales
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")]:
        net_move = (df["close"] - df["close"].shift(w)).abs()
        sum_moves = df["returns"].abs().rolling(w, min_periods=w//2).sum() * df["close"]
        df[f"efficiency_{label}"] = net_move / sum_moves.clip(lower=1e-10)

    # Parkinson volatility at multiple scales
    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    for w, label in [(60, "1h"), (240, "4h"), (1440, "24h")]:
        df[f"parkvol_{label}"] = np.sqrt((log_hl**2).rolling(w, min_periods=w//2).mean()) / np.sqrt(4 * np.log(2))

    # Kyle lambda rolling
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")]:
        df[f"kyle_lambda_{label}"] = df["kyle_lambda"].rolling(w, min_periods=w//2).mean()

    # Large trade imbalance rolling
    for w, label in [(5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")]:
        df[f"large_imb_{label}"] = df["large_imbalance"].rolling(w, min_periods=w//2).mean()

    # VWAP deviation z-score
    for w, label in [(60, "1h"), (240, "4h")]:
        vwap_roll = df["close_vs_vwap"].rolling(w, min_periods=w//2)
        df[f"vwap_zscore_{label}"] = (df["close_vs_vwap"] - vwap_roll.mean()) / vwap_roll.std().clip(lower=1e-10)

    # Trade count z-score
    for w, label in [(60, "1h"), (240, "4h"), (1440, "24h")]:
        tc_roll = df["trade_count"].rolling(w, min_periods=w//2)
        df[f"trade_count_zscore_{label}"] = (df["trade_count"] - tc_roll.mean()) / tc_roll.std().clip(lower=1e-10)

    print(f"    {len(df.columns)} columns after rolling", flush=True)

    # ---- 2. LAGGED FEATURES ----
    print("  [2/4] Lagged features (snapshots from 1h, 4h, 12h ago)...", flush=True)

    # Key features to lag
    lag_features = [
        "rvol_1h", "rvol_4h",
        "vol_zscore_1h", "vol_zscore_4h",
        "price_zscore_1h", "price_zscore_4h",
        "cum_imbalance_1h", "cum_imbalance_4h",
        "efficiency_1h", "efficiency_4h",
        "kyle_lambda_1h",
        "range_zscore_1h",
        "mom_1h", "mom_4h",
        "parkvol_1h",
        "trade_count_zscore_1h",
    ]

    for feat in lag_features:
        if feat not in df.columns:
            continue
        for lag, lag_label in [(60, "lag_1h"), (240, "lag_4h"), (720, "lag_12h")]:
            df[f"{feat}_{lag_label}"] = df[feat].shift(lag)

    print(f"    {len(df.columns)} columns after lags", flush=True)

    # ---- 3. RATE-OF-CHANGE FEATURES ----
    print("  [3/4] Rate-of-change features (deltas between horizons)...", flush=True)

    # Vol regime change: is vol increasing or decreasing?
    df["rvol_change_1h"] = df["rvol_1h"] - df["rvol_1h"].shift(60)
    df["rvol_change_4h"] = df["rvol_4h"] - df["rvol_4h"].shift(240)
    df["rvol_ratio_short_long"] = df["rvol_5m"] / df["rvol_24h"].clip(lower=1e-10)
    df["rvol_ratio_1h_24h"] = df["rvol_1h"] / df["rvol_24h"].clip(lower=1e-10)

    # Imbalance acceleration
    df["imbalance_accel_1h"] = df["cum_imbalance_1h"] - df["cum_imbalance_1h"].shift(60)
    df["imbalance_accel_4h"] = df["cum_imbalance_4h"] - df["cum_imbalance_4h"].shift(240)

    # Efficiency change (market becoming more/less trending)
    df["efficiency_change_1h"] = df["efficiency_1h"] - df["efficiency_1h"].shift(60)
    df["efficiency_change_4h"] = df["efficiency_4h"] - df["efficiency_4h"].shift(240)

    # Volume surge: current vs recent average
    df["vol_surge_vs_1h"] = df["volume"] / df["vol_mean_1h"].clip(lower=1e-10)
    df["vol_surge_vs_4h"] = df["volume"] / df["vol_mean_4h"].clip(lower=1e-10)

    # Kyle lambda change (price impact changing)
    df["kyle_change_1h"] = df["kyle_lambda_1h"] - df["kyle_lambda_1h"].shift(60)

    # Momentum acceleration
    df["mom_accel_1h"] = df["mom_1h"] - df["mom_1h"].shift(60)
    df["mom_accel_4h"] = df["mom_4h"] - df["mom_4h"].shift(240)

    # Price z-score change (moving toward or away from mean)
    df["zscore_change_1h"] = df["price_zscore_1h"] - df["price_zscore_1h"].shift(60)
    df["zscore_change_4h"] = df["price_zscore_4h"] - df["price_zscore_4h"].shift(240)

    print(f"    {len(df.columns)} columns after rate-of-change", flush=True)

    # ---- 4. CROSS-FEATURE INTERACTIONS ----
    print("  [4/4] Cross-feature interactions...", flush=True)

    # Vol × imbalance (high vol + strong imbalance = directional conviction)
    df["vol_x_imbalance_1h"] = df["rvol_1h"] * df["cum_imbalance_1h"]
    df["vol_x_imbalance_4h"] = df["rvol_4h"] * df["cum_imbalance_4h"]

    # Efficiency × vol (trending + volatile = strong trend)
    df["efficiency_x_vol_1h"] = df["efficiency_1h"] * df["rvol_1h"]

    # Price z-score × efficiency (extended + trending = continuation; extended + choppy = reversal)
    df["zscore_x_efficiency_1h"] = df["price_zscore_1h"] * df["efficiency_1h"]
    df["zscore_x_efficiency_4h"] = df["price_zscore_4h"] * df["efficiency_4h"]

    # Volume surge × imbalance (volume spike with directional bias)
    df["surge_x_imbalance"] = df["vol_surge_vs_1h"] * df["vol_imbalance"]

    # Range expansion × body (big range + big body = conviction; big range + small body = indecision)
    df["range_x_body"] = df["range_zscore_1h"] * df["body_pct"]

    elapsed = time.time() - t0
    n_features = len([c for c in df.columns if not c.startswith("tgt_")
                      and c not in ("open", "high", "low", "close", "volume",
                                    "timestamp_us", "returns")])
    print(f"  Feature engineering done: {n_features} features in {elapsed:.0f}s", flush=True)
    return df


# ============================================================
# TARGETS (same as before)
# ============================================================
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

    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    print(f"  Added {len(tgt_cols)} targets", flush=True)
    return df


# ============================================================
# PREDICTABILITY SCAN
# ============================================================
def select_top_features(df_train, target_col, feat_cols, top_n=30):
    y = df_train[target_col].values
    valid_y = np.isfinite(y)
    corrs = []
    for f in feat_cols:
        x = df_train[f].values
        mask = valid_y & np.isfinite(x)
        if mask.sum() < 500:
            corrs.append(0.0)
            continue
        try:
            c_val, _ = spearmanr(x[mask], y[mask])
            corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
        except:
            corrs.append(0.0)
    top_idx = np.argsort(corrs)[::-1][:top_n]
    selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]
    top_corrs = [(feat_cols[j], corrs[j]) for j in top_idx[:10]]
    return selected, top_corrs


def run_scan(df, feat_cols):
    """Run predictability scan: train/test split + WFO."""
    tgt_cols = sorted([c for c in df.columns if c.startswith("tgt_")])

    n = len(df)
    split = int(n * 0.75)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Targets:  {len(tgt_cols)}")
    print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}")

    results = []

    # Key targets to focus on
    focus_targets = [
        "tgt_profitable_long_3", "tgt_profitable_short_3",
        "tgt_profitable_long_5", "tgt_profitable_short_5",
        "tgt_profitable_long_10", "tgt_profitable_short_10",
        "tgt_mean_revert_5", "tgt_mean_revert_15", "tgt_mean_revert_30",
        "tgt_breakout_up_3", "tgt_breakout_down_3",
        "tgt_vol_expansion_5", "tgt_vol_expansion_15", "tgt_vol_expansion_30",
        "tgt_ret_mag_1", "tgt_ret_mag_3", "tgt_ret_mag_5",
        "tgt_ret_1", "tgt_ret_3", "tgt_ret_5", "tgt_ret_10",
    ]

    print(f"\n  {'Target':<35} {'Score':>8} {'Prev':>8} {'Delta':>8} {'Top Feature':<30}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*30}")

    # Previous scores (from basic features)
    prev_scores = {
        "tgt_profitable_long_3": 0.0641, "tgt_profitable_short_3": 0.0772,
        "tgt_profitable_long_5": 0.0493, "tgt_profitable_short_5": 0.0598,
        "tgt_profitable_long_10": 0.0322, "tgt_profitable_short_10": 0.0354,
        "tgt_mean_revert_5": 0.1924, "tgt_mean_revert_15": 0.1710,
        "tgt_mean_revert_30": 0.1585,
        "tgt_breakout_up_3": 0.2399, "tgt_breakout_down_3": 0.2318,
        "tgt_vol_expansion_5": 0.1427, "tgt_vol_expansion_15": 0.1661,
        "tgt_vol_expansion_30": 0.1866,
        "tgt_ret_mag_1": 0.4075, "tgt_ret_mag_3": 0.4050, "tgt_ret_mag_5": 0.3994,
        "tgt_ret_1": -0.0036, "tgt_ret_3": 0.0018, "tgt_ret_5": 0.0096,
        "tgt_ret_10": 0.0152,
    }

    for tgt in focus_targets:
        if tgt not in df.columns:
            continue

        y_train = df_train[tgt].values
        y_test = df_test[tgt].values
        valid_tr = np.isfinite(y_train)
        valid_te = np.isfinite(y_test)

        if valid_tr.sum() < 1000 or valid_te.sum() < 500:
            continue

        unique_vals = np.unique(y_train[valid_tr])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0})

        selected, top_corrs = select_top_features(df_train, tgt, feat_cols)
        if len(selected) < 5:
            continue

        X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
        y_tr = y_train[valid_tr]
        y_te = y_test[valid_te]

        try:
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective="binary", metric="auc", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr.astype(int))
                pred = model.predict_proba(X_te)[:, 1]
                score = roc_auc_score(y_te.astype(int), pred) - 0.5
            else:
                model = lgb.LGBMRegressor(
                    objective="regression", metric="rmse", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                s, _ = spearmanr(pred, y_te)
                score = s if np.isfinite(s) else 0.0

            prev = prev_scores.get(tgt, 0.0)
            delta = score - prev
            top_feat = top_corrs[0][0] if top_corrs else "?"

            marker = "+++" if delta > 0.02 else "++" if delta > 0.01 else "+" if delta > 0 else ""
            print(f"  {tgt:<35} {score:>+.4f} {prev:>+.4f} {delta:>+.4f} {top_feat:<30} {marker}")

            results.append({
                "target": tgt, "score": score, "prev": prev, "delta": delta,
                "is_binary": is_binary, "top_feat": top_feat,
                "top_corrs": top_corrs,
            })
        except Exception as e:
            print(f"  {tgt:<35} ERROR: {e}")

    return results


# ============================================================
# P&L TEST (expanded)
# ============================================================
def pnl_test(df, feat_cols):
    """P&L test with expanded features, longer hold periods too."""
    n = len(df)
    test_bars = 60 * 24 * 60  # 60 days
    train_end = n - test_bars
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]

    print(f"\n  P&L Test: train={len(df_train):,}, test={len(df_test):,}")

    # Test multiple hold periods
    for hold, tgt_suffix in [(3, "3"), (5, "5"), (10, "10")]:
        print(f"\n  === HOLD {hold} BARS ({hold}m) ===")

        for direction, tgt_name in [("LONG", f"tgt_profitable_long_{tgt_suffix}"),
                                     ("SHORT", f"tgt_profitable_short_{tgt_suffix}")]:
            if tgt_name not in df.columns:
                continue

            y_tr = df_train[tgt_name].values
            valid_tr = np.isfinite(y_tr)

            selected, _ = select_top_features(df_train, tgt_name, feat_cols)
            if len(selected) < 5:
                continue

            X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
            y_tr_c = y_tr[valid_tr].astype(int)

            model = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=20, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
            model.fit(X_tr, y_tr_c)

            y_te = df_test[tgt_name].values
            valid_te = np.isfinite(y_te)
            X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
            pred = model.predict_proba(X_te)[:, 1]

            fwd_ret = df_test[f"tgt_ret_{tgt_suffix}"].values[valid_te]

            print(f"\n  {direction} (hold={hold}m):")
            print(f"  {'Thresh':>8} {'Trades':>8} {'WR':>8} {'AvgRet':>10} {'Total':>10} {'Sharpe':>8}")
            print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                mask = pred >= thresh
                if mask.sum() < 20:
                    continue

                if direction == "LONG":
                    trade_rets = fwd_ret[mask] - FEE_FRAC
                else:
                    trade_rets = -fwd_ret[mask] - FEE_FRAC

                trade_rets = trade_rets[np.isfinite(trade_rets)]
                if len(trade_rets) < 10:
                    continue

                n_trades = len(trade_rets)
                wr = (trade_rets > 0).mean()
                avg = trade_rets.mean()
                total = trade_rets.sum()
                ann_factor = np.sqrt(252 * 24 * 60 / hold)
                sharpe = avg / trade_rets.std() * ann_factor if trade_rets.std() > 0 else 0

                flag = " *" if avg > 0 else ""
                print(f"  {thresh:>8.2f} {n_trades:>8} {wr:>8.1%} "
                      f"{avg*10000:>+10.2f}bp {total*100:>+10.2f}% {sharpe:>+8.1f}{flag}")

    # Mean reversion P&L with longer holds
    print(f"\n  === MEAN REVERSION (contrarian, various holds) ===")
    ma_1h = df_test["close"].rolling(60).mean().values

    for hold, tgt_suffix in [(5, "5"), (15, "15"), (30, "30")]:
        tgt_mr = f"tgt_mean_revert_{tgt_suffix}"
        if tgt_mr not in df.columns:
            continue

        y_tr = df_train[tgt_mr].values
        valid_tr = np.isfinite(y_tr)
        selected, _ = select_top_features(df_train, tgt_mr, feat_cols)
        if len(selected) < 5:
            continue

        X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X_tr, y_tr[valid_tr].astype(int))

        y_te = df_test[tgt_mr].values
        valid_te = np.isfinite(y_te)
        X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
        pred = model.predict_proba(X_te)[:, 1]

        close_test = df_test["close"].values[valid_te]
        above_ma = close_test > ma_1h[valid_te]

        fwd_ret_col = f"tgt_ret_{tgt_suffix}" if f"tgt_ret_{tgt_suffix}" in df_test.columns else "tgt_ret_5"
        fwd_ret = df_test[fwd_ret_col].values[valid_te]

        print(f"\n  Mean Revert (hold={hold}m):")
        print(f"  {'Thresh':>8} {'Trades':>8} {'WR':>8} {'AvgRet':>10} {'Total':>10} {'Sharpe':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

        for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
            mask = pred >= thresh
            if mask.sum() < 20:
                continue

            trade_rets = np.where(above_ma[mask], -fwd_ret[mask], fwd_ret[mask]) - FEE_FRAC
            trade_rets = trade_rets[np.isfinite(trade_rets)]
            if len(trade_rets) < 10:
                continue

            n_trades = len(trade_rets)
            wr = (trade_rets > 0).mean()
            avg = trade_rets.mean()
            total = trade_rets.sum()
            ann_factor = np.sqrt(252 * 24 * 60 / hold)
            sharpe = avg / trade_rets.std() * ann_factor if trade_rets.std() > 0 else 0

            flag = " *" if avg > 0 else ""
            print(f"  {thresh:>8.2f} {n_trades:>8} {wr:>8.1%} "
                  f"{avg*10000:>+10.2f}bp {total*100:>+10.2f}% {sharpe:>+8.1f}{flag}")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  1-MINUTE EXPANDED FEATURES TEST")
    print(f"  {SYMBOL}, {START_DATE} to {END_DATE}")
    print(f"  Longer horizons (4h/12h/24h) + lags + rate-of-change + interactions")
    print("=" * 80)

    df = load_1m_bars()

    print("\n  Building expanded features...", flush=True)
    df = add_expanded_features(df)

    print("  Adding targets...", flush=True)
    df = add_targets(df)

    # Drop warmup (24h = 1440 bars to let all rolling features stabilize)
    df = df.iloc[1440:].copy()
    print(f"  After warmup: {len(df):,} bars")

    feat_cols = sorted([c for c in df.columns
                        if not c.startswith("tgt_")
                        and c not in ("open", "high", "low", "close", "volume",
                                      "timestamp_us", "returns")])

    print(f"  Total features: {len(feat_cols)}")

    # Part 1: Predictability scan
    print(f"\n{'=' * 80}")
    print("  PART 1: PREDICTABILITY SCAN (expanded vs basic features)")
    print("=" * 80)
    results = run_scan(df, feat_cols)

    # Part 2: Feature importance for improved targets
    print(f"\n{'=' * 80}")
    print("  PART 2: TOP FEATURES FOR KEY TARGETS")
    print("=" * 80)

    for r in results:
        if r["delta"] > 0.005:
            print(f"\n  {r['target']} (score={r['score']:+.4f}, delta={r['delta']:+.4f}):")
            for fname, corr in r["top_corrs"][:8]:
                new_flag = " [NEW]" if any(x in fname for x in ["lag_", "change_", "accel_",
                                           "surge_", "_x_", "12h", "24h"]) else ""
                print(f"    {corr:.4f}  {fname}{new_flag}")

    # Part 3: P&L test
    print(f"\n{'=' * 80}")
    print("  PART 3: P&L TEST (expanded features)")
    print("=" * 80)
    pnl_test(df, feat_cols)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
